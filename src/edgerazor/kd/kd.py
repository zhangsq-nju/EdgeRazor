"""
Knowledge Distillation (KD) module implementation for EdgeRazor.
- total_loss = loss_task_alpha * task_loss + distill_loss
- distill_loss = alpha_1 * loss_1 + alpha_2 * loss_2 + ... + alpha_n * loss_n

Model forward details:
model_inputs:
- input_ids: torch.Size([1, seq_len])
- attention_mask: torch.Size([1, seq_len])
- labels: torch.Size([1, seq_len])
```
output = model(
    **model_inputs,
    return_dict=True,
    output_hidden_states=True,
    output_attentions=True,
)
```

Based on model output:
- transformers.modeling_outputs.CausalLMOutputWithPast
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
- transformers.modeling_outputs.MoeCausalLMOutputWithPast
    loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[tuple[torch.FloatTensor]] = None

Details:
- output['loss']: Task-specific loss (e.g., CrossEntropyLoss)
- output['logits'].shape = (batch_size, seq_len, vocab_size)
- output['past_key_values'][decoder_layer_index] = (key_states, value_states)
  - key_states.shape = (batch_size, num_key_value_heads, seq_length, head_dim=hidden_size/num_attention_heads)
  - value_states.shape = (batch_size, num_key_value_heads, seq_length, head_dim=hidden_size/num_attention_heads)
- output['hidden_states'][layer_index].shape = (batch_size, seq_len, hidden_size)
- output['attentions'][decoder_layer_index].shape = (batch_size, num_heads, query_seq_len, key_seq_len)

API format:
- compute_loss(student_outputs, teacher_outputs, labels): calculate total_loss, arrange all distill losses
  - student_outputs: model output (dict or ModelOutput) with 'loss' field containing task_loss
  - teacher_outputs: model output (dict, ModelOutput, or Tensor) for distillation
  - labels: ground truth labels
  - kd_config: loaded in __init__, contains all loss_i configurations
- compute_xxx(student_inputs, teacher_inputs, labels, kd_config_loss): calculate individual distill loss
  - student_inputs/teacher_inputs: logits, hidden_states, attentions, past_key_values, etc.
  - labels: ground truth labels
  - kd_config_loss: LossConfig object containing all parameters (alpha, temperature, padding_id, etc.)

Distill function format: `compute_xxx(...)`
- kldf: Kullback-Leibler Divergence Forward
- kldr: Kullback-Leibler Divergence Reverse
- kldc: Kullback-Leibler Divergence Confidence
- fd: Feature Distillation (MSE Loss)
"""
# ruff: noqa: N812

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache

from ..log import get_logger
from .util import DistillConfig, get_distill_function
from .util.layer_select import resolve_layer_indices, resolve_layer_indices_adaptive


class KD:
    """
    Knowledge Distillation (KD) class for EdgeRazor.
    
    Implements knowledge distillation with flexible multi-loss configuration.
    Formula: total_loss = loss_task_alpha * task_loss + distill_loss
             distill_loss = alpha_1 * loss_1 + alpha_2 * loss_2 + ... + alpha_n * loss_n
    
    Args:
        config: Configuration for KD, can be:
            - DistillConfig object
            - dict: Python dictionary
            - str/Path: Path to YAML/JSON configuration file
    
    Examples:
        >>> # From YAML file
        >>> kd = KD("configs/kd_kldc_fd.yaml")
        >>>
        >>> # From dict
        >>> kd = KD({
        ...     "method": "KD",
        ...     "loss_1": {
        ...         "loss_type": "logits",
        ...         "loss_function": "kldc",
        ...         "alpha": 0.7,
        ...         "temperature": 2.0
        ...     }
        ... })
        >>>
        >>> # Compute loss
        >>> total_loss, loss_dict = kd.compute_loss(
        ...     student_outputs=student_outputs,
        ...     teacher_outputs=teacher_outputs,
        ...     labels=labels
        ... )
    """
    
    def __init__(self, config):
        """
        Initialize KD with configuration.
        
        Args:
            config: Configuration (DistillConfig, dict, or file path)
        """
        self.logger = get_logger('KD')
        self.logger.info('Initializing Knowledge Distillation (KD)')
        
        # Load configuration
        self.config = self._load_configuration(config)
        self._log_configuration()
        
        # Initialize loss functions
        self.loss_functions = {}
        for loss_key, loss_config in self.config.losses.items():
            loss_fn = get_distill_function(loss_config.loss_function)
            self.loss_functions[loss_key] = loss_fn
            self.logger.debug(
                f'Registered {loss_key}: '
                f'type={loss_config.loss_type}, '
                f'function={loss_config.loss_function}, '
                f'alpha={loss_config.alpha}'
            )
        
        self.logger.info('KD initialization completed')
    
    def _load_configuration(self, config):
        """
        Load configuration from various formats.
        
        Args:
            config: DistillConfig object, dict, or file path
        
        Returns:
            DistillConfig: Loaded configuration
        
        Raises:
            ValueError: If file format is unsupported
            TypeError: If config type is invalid
        """
        try:
            # DistillConfig object
            if isinstance(config, DistillConfig):
                self.logger.info('Using provided DistillConfig object')
                return config

            # Python dictionary
            elif isinstance(config, dict):
                self.logger.info('Loading configuration from dictionary')
                return DistillConfig.from_dict(config)

            # File path (YAML or JSON)
            elif isinstance(config, (str, Path)):
                config_path = Path(config)
                self.logger.info(f'Loading configuration from: {config_path}')

                suffix = config_path.suffix.lower()
                if suffix in ['.yaml', '.yml']:
                    self.logger.debug('Configuration loaded from YAML file')
                    return DistillConfig.from_yaml(config_path)
                elif suffix == '.json':
                    self.logger.debug('Configuration loaded from JSON file')
                    return DistillConfig.from_json(config_path)
                else:
                    raise ValueError(
                        f'Unsupported file format: {suffix}. '
                        f'Supported formats: .yaml, .yml, .json'
                    )
            
            # Invalid type
            else:
                raise TypeError(
                    f'Invalid config type: {type(config).__name__}. '
                    f'Expected: DistillConfig, dict, str, or Path'
                )
        
        except Exception as e:
            self.logger.error(f'Failed to load configuration: {e}')
            raise
    
    def _log_configuration(self):
        """Log configuration details at DEBUG level."""
        self.logger.debug('=' * 80)
        self.logger.debug('KD Configuration')
        self.logger.debug('=' * 80)
        self.logger.debug(f'Method: {self.config.method}')
        self.logger.debug(f'Task loss alpha: {self.config.loss_task_alpha}')
        self.logger.debug(f'Number of losses: {len(self.config.losses)}')

        for loss_key, loss_config in self.config.losses.items():
            self.logger.debug(f'{loss_key}:')
            self.logger.debug(f'  loss_type:     {loss_config.loss_type}')
            self.logger.debug(f'  loss_function: {loss_config.loss_function}')
            self.logger.debug(f'  alpha:         {loss_config.alpha}')

            if loss_config.loss_type == 'logits':
                self.logger.debug(f'  temperature:   {loss_config.temperature}')
                self.logger.debug(f'  use_entropy:   {loss_config.use_entropy}')

            self.logger.debug(f'  reduction:     {loss_config.reduction}')

        self.logger.debug('=' * 80)
    
    def compute_loss(
        self,
        student_outputs,
        teacher_outputs,
        labels
    ):
        """
        Compute total loss with knowledge distillation.

        Formula:
            total_loss = loss_task_alpha * task_loss + distill_loss
            distill_loss = alpha_1 * loss_1 + alpha_2 * loss_2 + ... + alpha_n * loss_n

        Args:
            student_outputs: Student model outputs, can be:
                - dict: {'loss': ..., 'logits': ..., 'hidden_states': ..., 'attentions': ...}
                - ModelOutput: transformers output object with 'loss' attribute
            teacher_outputs: Teacher model outputs, can be:
                - torch.Tensor: logits only
                - dict: {'logits': ..., 'hidden_states': ..., 'attentions': ...}
                - ModelOutput: transformers output object
            labels: Ground truth labels (torch.Tensor)

        Returns:
            tuple: (total_loss, loss_dict)
                - total_loss: torch.Tensor, sum of task_loss and distill_loss
                - loss_dict: dict with keys:
                    - 'task_loss': float, task-specific loss value
                    - 'distill_loss': float, total distillation loss (Σ alpha_i * loss_i)
                    - 'distill_loss_details': dict, individual loss values {'loss_1': float, 'loss_2': float, ...}
                    - 'total_loss': float, final total loss

        Examples:
            >>> # With ModelOutput
            >>> student_outputs = student_model(**inputs, labels=labels, return_dict=True)
            >>> teacher_outputs = teacher_model(**inputs, return_dict=True)
            >>> total_loss, loss_dict = kd.compute_loss(
            ...     student_outputs, teacher_outputs, labels
            ... )
            >>> # loss_dict = {
            >>> #     'task_loss': 2.5,
            >>> #     'distill_loss': 0.85,
            >>> #     'total_loss': 3.35,
            >>> #     'distill_loss_details': {'loss_1': 0.7, 'loss_2': 0.5},
            >>> # }
            >>>
            >>> # With dict outputs
            >>> student_outputs = {
            ...     'loss': task_loss,
            ...     'logits': student_logits,
            ...     'hidden_states': student_hidden_states
            ... }
            >>> teacher_outputs = {
            ...     'logits': teacher_logits,
            ...     'hidden_states': teacher_hidden_states
            ... }
            >>> total_loss, loss_dict = kd.compute_loss(
            ...     student_outputs, teacher_outputs, labels
            ... )
        """
        # Extract task loss from student outputs
        if isinstance(student_outputs, dict):
            task_loss = student_outputs.get('loss')
        else:
            # ModelOutput object
            task_loss = getattr(student_outputs, 'loss', None)

        if task_loss is None:
            raise ValueError(
                "task_loss not found in student_outputs. "
                "student_outputs must contain 'loss' (dict) or have 'loss' attribute (ModelOutput)."
            )

        # Convert outputs to dict format for unified handling
        if not isinstance(student_outputs, dict):
            student_outputs = {
                'loss': getattr(student_outputs, 'loss', None),
                'logits': getattr(student_outputs, 'logits', None),
                'hidden_states': getattr(student_outputs, 'hidden_states', None),
                'attentions': getattr(student_outputs, 'attentions', None),
                'past_key_values': getattr(student_outputs, 'past_key_values', None),
            }

        if isinstance(teacher_outputs, torch.Tensor):
            teacher_outputs = {'logits': teacher_outputs}
        elif not isinstance(teacher_outputs, dict):
            teacher_outputs = {
                'logits': getattr(teacher_outputs, 'logits', None),
                'hidden_states': getattr(teacher_outputs, 'hidden_states', None),
                'attentions': getattr(teacher_outputs, 'attentions', None),
                'past_key_values': getattr(teacher_outputs, 'past_key_values', None),
            }
        
        # Initialize loss dictionary
        loss_dict = {
            'total_loss': 0.0,
            'task_loss': task_loss.item() if isinstance(task_loss, torch.Tensor) else task_loss,
            'distill_loss': 0.0,
            'distill_loss_details': {},
        }
        
        # Compute distillation losses: distill_loss = Σ(alpha_i * loss_i)
        distill_loss = 0.0
        
        for loss_key, loss_config in self.config.losses.items():
            loss_fn = self.loss_functions[loss_key]
            
            # Logits distillation (KLD-based)
            if loss_config.loss_type == 'logits':
                student_logits = student_outputs.get('logits')
                teacher_logits = teacher_outputs.get('logits')
                
                if student_logits is None or teacher_logits is None:
                    self.logger.warning(
                        f'{loss_key}: logits not found in outputs, skipping'
                    )
                    continue
                
                loss_value = loss_fn(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    kd_config_loss=loss_config
                )
                
                weighted_loss = loss_config.alpha * float(loss_value)
                distill_loss += weighted_loss
                loss_dict['distill_loss_details'][loss_key] = float(loss_value)
            
            # Feature/Hidden states distillation (MSE-based)
            elif loss_config.loss_type == 'hidden_states':
                student_features = student_outputs.get('hidden_states')
                teacher_features = teacher_outputs.get('hidden_states')
                
                if student_features is None or teacher_features is None:
                    self.logger.warning(
                        f'{loss_key}: features/hidden_states not found in outputs, skipping'
                    )
                    continue
                
                # Handle layer selection
                # hidden_states can be:
                # - Single tensor: (batch_size, seq_len, hidden_size)
                # - Tuple of tensors: (layer_0, layer_1, ..., layer_n)
                #   where each layer_i has shape (batch_size, seq_len, hidden_size)
                
                if loss_config.layer_index is not None:
                    # If hidden_states is tuple, select specific layers
                    if isinstance(student_features, tuple):
                        # Convert layer_index to list for unified handling
                        if isinstance(loss_config.layer_index, int):
                            layer_indices = [loss_config.layer_index]
                        elif isinstance(loss_config.layer_index, str):
                            layer_indices = [loss_config.layer_index]
                        else:
                            layer_indices = loss_config.layer_index
                        
                        # Get total number of layers
                        num_layers = len(student_features)
                        
                        # Resolve string layer names to actual indices
                        # If fixed layer_indices are used,
                        if "adaptive" not in layer_indices:
                            resolved_indices = resolve_layer_indices(
                                layer_indices=layer_indices,
                                num_layers=num_layers,
                                loss_key=loss_key,
                                logger=self.logger,
                            )
                        # If adaptive layer selection is used, calculate cosine similarity to select layers
                        elif "adaptive" in layer_indices:
                            resolved_indices = resolve_layer_indices_adaptive(
                                hidden_states=student_features,
                                metric=loss_config.layer_index_adaptive_metric,
                                topk=loss_config.layer_index_adaptive_topk,
                            )
                        
                        # Compute loss for each selected layer and accumulate
                        layer_loss = 0.0
                        num_valid_layers = 0
                        
                        for actual_idx in resolved_indices:
                            if actual_idx < 0 or actual_idx >= num_layers:
                                self.logger.warning(
                                    f'{loss_key}: layer_index {actual_idx} out of range '
                                    f'(total {num_layers} layers), skipping this layer'
                                )
                                continue
                            
                            if actual_idx >= len(teacher_features):
                                self.logger.warning(
                                    f'{loss_key}: teacher layer {actual_idx} out of range '
                                    f'(total {len(teacher_features)} layers), skipping this layer'
                                )
                                continue
                            
                            # Compute loss for this layer
                            loss_value = loss_fn(
                                student_features=student_features[actual_idx],
                                teacher_features=teacher_features[actual_idx],
                                labels=labels,
                                kd_config_loss=loss_config
                            )
                            
                            layer_loss += float(loss_value)
                            num_valid_layers += 1
                        
                        if num_valid_layers == 0:
                            self.logger.warning(
                                f'{loss_key}: no valid layers found for distillation, skipping'
                            )
                            continue
                        
                        # Average loss across selected layers
                        loss_value = layer_loss / num_valid_layers
                    else:
                        # If hidden_states is a single tensor, ignore layer_index
                        self.logger.warning(
                            f'{loss_key}: layer_index specified but hidden_states is not a tuple, '
                            f'using the single tensor for distillation'
                        )
                        loss_value = loss_fn(
                            student_features=student_features,
                            teacher_features=teacher_features,
                            labels=labels,
                            kd_config_loss=loss_config
                        )
                else:
                    # No layer_index specified, use all features
                    loss_value = loss_fn(
                        student_features=student_features,
                        teacher_features=teacher_features,
                        labels=labels,
                        kd_config_loss=loss_config
                    )
                
                weighted_loss = loss_config.alpha * float(loss_value)
                distill_loss += weighted_loss
                loss_dict['distill_loss_details'][loss_key] = float(loss_value)
            
            # Attention distillation (KLD-based on attention distributions)
            elif loss_config.loss_type == 'attentions':
                student_attentions = student_outputs.get('attentions')
                teacher_attentions = teacher_outputs.get('attentions')
                
                # Attention data structure:
                # Tuple of length num_layers,
                # each element is of shape (batch_size, num_attention_heads, seq_len, seq_len)
                # The last dimension represents attention distribution over keys for each query
                # KLD/MSE loss is typically applied on the last dimension (attention weights)
                
                if student_attentions is None or teacher_attentions is None:
                    self.logger.warning(
                        f'{loss_key}: attentions not found in outputs, skipping'
                    )
                    continue
                
                if loss_config.layer_index is not None:
                    # If attentions is tuple, select specific layers
                    if isinstance(student_attentions, tuple):
                        # Convert layer_index to list for unified handling
                        if isinstance(loss_config.layer_index, int):
                            layer_indices = [loss_config.layer_index]
                        elif isinstance(loss_config.layer_index, str):
                            layer_indices = [loss_config.layer_index]
                        else:
                            layer_indices = loss_config.layer_index
                        
                        # Get total number of layers
                        num_layers = len(student_attentions)
                        
                        # Resolve string layer names to actual indices
                        if "adaptive" not in layer_indices:
                            resolved_indices = resolve_layer_indices(
                                layer_indices=layer_indices,
                                num_layers=num_layers,
                                loss_key=loss_key,
                                logger=self.logger,
                            )
                        # Adaptive layer selection based on attention patterns
                        elif "adaptive" in layer_indices:
                            # For attention, we can use the attention matrices directly
                            # Flatten attention to (batch_size, num_heads * seq_len * seq_len) for similarity
                            resolved_indices = resolve_layer_indices_adaptive(
                                hidden_states=student_attentions,
                                metric=loss_config.layer_index_adaptive_metric,
                                topk=loss_config.layer_index_adaptive_topk,
                            )
                        
                        # Compute loss for each selected layer and accumulate
                        layer_loss = 0.0
                        num_valid_layers = 0
                        
                        for actual_idx in resolved_indices:
                            if actual_idx < 0 or actual_idx >= num_layers:
                                self.logger.warning(
                                    f'{loss_key}: layer_index {actual_idx} out of range '
                                    f'(total {num_layers} layers), skipping this layer'
                                )
                                continue
                            
                            if actual_idx >= len(teacher_attentions):
                                self.logger.warning(
                                    f'{loss_key}: teacher layer {actual_idx} out of range '
                                    f'(total {len(teacher_attentions)} layers), skipping this layer'
                                )
                                continue
                            
                            # Get attention matrices for this layer
                            # Shape: (batch_size, num_attention_heads, seq_len, seq_len)
                            student_attn = student_attentions[actual_idx]
                            teacher_attn = teacher_attentions[actual_idx]
                            
                            # Handle head dimension mismatch if student has fewer heads
                            if student_attn.shape[1] != teacher_attn.shape[1]:
                                # Average teacher heads to match student head count
                                teacher_num_heads = teacher_attn.shape[1]
                                student_num_heads = student_attn.shape[1]
                                
                                if teacher_num_heads > student_num_heads:
                                    # Group teacher heads and average
                                    heads_per_group = teacher_num_heads // student_num_heads
                                    teacher_attn = teacher_attn.view(
                                        teacher_attn.shape[0],
                                        student_num_heads,
                                        heads_per_group,
                                        teacher_attn.shape[2],
                                        teacher_attn.shape[3]
                                    ).mean(dim=2)
                                else:
                                    self.logger.warning(
                                        f'{loss_key}: student has more attention heads than teacher, '
                                        f'this is unusual, skipping layer {actual_idx}'
                                    )
                                    continue
                            
                            # Compute loss for this layer
                            # The loss function should handle attention matrices
                            # Attention weights are already normalized (softmax applied)
                            loss_value = loss_fn(
                                student_logits=student_attn,
                                teacher_logits=teacher_attn,
                                labels=labels,
                                kd_config_loss=loss_config
                            )
                            
                            layer_loss += float(loss_value)
                            num_valid_layers += 1
                        
                        if num_valid_layers == 0:
                            self.logger.warning(
                                f'{loss_key}: no valid layers found for attention distillation, skipping'
                            )
                            continue
                        
                        # Average loss across selected layers
                        loss_value = layer_loss / num_valid_layers
                    else:
                        # If attentions is a single tensor, use it directly
                        self.logger.warning(
                            f'{loss_key}: layer_index specified but attentions is not a tuple, '
                            f'using the single tensor for distillation'
                        )
                        loss_value = loss_fn(
                            student_logits=student_attentions,
                            teacher_logits=teacher_attentions,
                            labels=labels,
                            kd_config_loss=loss_config
                        )
                else:
                    # No layer_index specified, compute loss across all layers
                    # Combine all tuple elements into single tensors for loss computation
                    student_attentions_all = torch.stack(student_attentions, dim=0)
                    teacher_attentions_all = torch.stack(teacher_attentions, dim=0)
                    loss_value = loss_fn(
                        student_logits=student_attentions_all,
                        teacher_logits=teacher_attentions_all,
                        labels=labels,
                        kd_config_loss=loss_config
                    )
                
                weighted_loss = loss_config.alpha * float(loss_value)
                distill_loss += weighted_loss
                loss_dict['distill_loss_details'][loss_key] = float(loss_value)
            
            # Past key values distillation (Value-Value relation based)
            elif loss_config.loss_type == 'past_key_values':
                student_past = student_outputs.get('past_key_values')
                teacher_past = teacher_outputs.get('past_key_values')
                
                # past_key_values data structure:
                # transformers.cache_utils.DynamicCache or tuple of length num_layers,
                # each element is a tuple of (key_states, value_states)
                # each of shape (batch_size, num_key_value_heads, seq_len, head_dim)
                #
                # For value-value relation distillation:
                # V @ V^T => (batch_size, num_key_value_heads, seq_len, seq_len)
                # This captures the similarity between different positions in the sequence
                # KLD/MSE loss is applied on the last dimension after softmax normalization
                
                if student_past is None:
                    self.logger.warning(
                        f'{loss_key}: student past_key_values not found in outputs, skipping'
                    )
                    continue
                
                if teacher_past is None:
                    self.logger.warning(
                        f'{loss_key}: teacher past_key_values not found in outputs, skipping'
                    )
                    continue
                
                # Determine which component to use: 'key', 'value', or 'both'
                kv_component = loss_config.self_relation_dsitill_component
                
                if loss_config.layer_index is not None:
                    # If past_key_values is Cache, select specific layers
                    if isinstance(student_past, Cache):
                        # Convert layer_index to list for unified handling
                        if isinstance(loss_config.layer_index, int):
                            layer_indices = [loss_config.layer_index]
                        elif isinstance(loss_config.layer_index, str):
                            layer_indices = [loss_config.layer_index]
                        else:
                            layer_indices = loss_config.layer_index
                        
                        # Get total number of layers
                        num_layers = len(student_past)
                        
                        # Resolve string layer names to actual indices
                        if "adaptive" not in layer_indices:
                            resolved_indices = resolve_layer_indices(
                                layer_indices=layer_indices,
                                num_layers=num_layers,
                                loss_key=loss_key,
                                logger=self.logger,
                            )
                        # Adaptive layer selection based on key/value patterns
                        elif "adaptive" in layer_indices:
                            # Extract value states for adaptive selection
                            value_states = tuple(kv[1] for kv in student_past)
                            resolved_indices = resolve_layer_indices_adaptive(
                                hidden_states=value_states,
                                metric=loss_config.layer_index_adaptive_metric,
                                topk=loss_config.layer_index_adaptive_topk,
                            )
                        
                        # Compute loss for each selected layer and accumulate
                        layer_loss = 0.0
                        num_valid_layers = 0
                        
                        for actual_idx in resolved_indices:
                            if actual_idx < 0 or actual_idx >= num_layers:
                                self.logger.warning(
                                    f'{loss_key}: layer_index {actual_idx} out of range '
                                    f'(total {num_layers} layers), skipping this layer'
                                )
                                continue
                            
                            if actual_idx >= len(teacher_past):
                                self.logger.warning(
                                    f'{loss_key}: teacher layer {actual_idx} out of range '
                                    f'(total {len(teacher_past)} layers), skipping this layer'
                                )
                                continue
                            
                            # Get key and value states for this layer
                            # Each element is (key_states, value_states)
                            # Shape: (batch_size, num_key_value_heads, seq_len, head_dim)
                            student_kv = student_past[actual_idx]
                            teacher_kv = teacher_past[actual_idx]
                            
                            # Extract key and value states
                            student_keys, student_values = student_kv[0], student_kv[1]
                            teacher_keys, teacher_values = teacher_kv[0], teacher_kv[1]
                            
                            # Compute relation matrices based on kv_component setting
                            if kv_component == 'value' or kv_component == 'both':
                                # Compute V @ V^T for value-value relations
                                # Shape: (batch_size, num_kv_heads, seq_len, seq_len)
                                student_vv = torch.matmul(
                                    student_values, student_values.transpose(-1, -2)
                                )
                                teacher_vv = torch.matmul(
                                    teacher_values, teacher_values.transpose(-1, -2)
                                )
                                
                                # Scale by sqrt(head_dim) for numerical stability
                                head_dim = student_values.shape[-1]
                                student_vv = student_vv / (head_dim ** 0.5)
                                teacher_vv = teacher_vv / (head_dim ** 0.5)
                                
                                # Apply softmax to convert attention relations to probability distributions
                                # Normalize over the key dimension (last dimension)
                                student_vv = F.softmax(student_vv, dim=-1)
                                teacher_vv = F.softmax(teacher_vv, dim=-1)
                                
                                # Handle head count mismatch
                                if student_vv.shape[1] != teacher_vv.shape[1]:
                                    teacher_num_heads = teacher_vv.shape[1]
                                    student_num_heads = student_vv.shape[1]
                                    
                                    if teacher_num_heads > student_num_heads:
                                        heads_per_group = teacher_num_heads // student_num_heads
                                        teacher_vv = teacher_vv.view(
                                            teacher_vv.shape[0],
                                            student_num_heads,
                                            heads_per_group,
                                            teacher_vv.shape[2],
                                            teacher_vv.shape[3]
                                        ).mean(dim=2)
                                    else:
                                        self.logger.warning(
                                            f'{loss_key}: student has more kv heads than teacher, '
                                            f'skipping layer {actual_idx}'
                                        )
                                        continue
                            
                            if kv_component == 'key' or kv_component == 'both':
                                # Compute K @ K^T for key-key relations
                                # Shape: (batch_size, num_kv_heads, seq_len, seq_len)
                                student_kk = torch.matmul(
                                    student_keys, student_keys.transpose(-1, -2)
                                )
                                teacher_kk = torch.matmul(
                                    teacher_keys, teacher_keys.transpose(-1, -2)
                                )
                                
                                # Scale by sqrt(head_dim) for numerical stability
                                head_dim = student_keys.shape[-1]
                                student_kk = student_kk / (head_dim ** 0.5)
                                teacher_kk = teacher_kk / (head_dim ** 0.5)
                                
                                # Apply softmax to convert attention relations to probability distributions
                                # Normalize over the key dimension (last dimension)
                                student_kk = F.softmax(student_kk, dim=-1)
                                teacher_kk = F.softmax(teacher_kk, dim=-1)
                                
                                # Handle head count mismatch
                                if student_kk.shape[1] != teacher_kk.shape[1]:
                                    teacher_num_heads = teacher_kk.shape[1]
                                    student_num_heads = student_kk.shape[1]
                                    
                                    if teacher_num_heads > student_num_heads:
                                        heads_per_group = teacher_num_heads // student_num_heads
                                        teacher_kk = teacher_kk.view(
                                            teacher_kk.shape[0],
                                            student_num_heads,
                                            heads_per_group,
                                            teacher_kk.shape[2],
                                            teacher_kk.shape[3]
                                        ).mean(dim=2)
                                    else:
                                        self.logger.warning(
                                            f'{loss_key}: student has more kv heads than teacher, '
                                            f'skipping layer {actual_idx}'
                                        )
                                        continue
                            
                            # Compute loss based on component selection
                            if kv_component == 'value':
                                loss_value = loss_fn(
                                    student_logits=student_vv,
                                    teacher_logits=teacher_vv,
                                    labels=labels,
                                    kd_config_loss=loss_config
                                )
                            elif kv_component == 'key':
                                loss_value = loss_fn(
                                    student_logits=student_kk,
                                    teacher_logits=teacher_kk,
                                    labels=labels,
                                    kd_config_loss=loss_config
                                )
                            elif kv_component == 'both':
                                # Average loss from both key and value relations
                                loss_value_vv = loss_fn(
                                    student_logits=student_vv,
                                    teacher_logits=teacher_vv,
                                    labels=labels,
                                    kd_config_loss=loss_config
                                )
                                loss_value_kk = loss_fn(
                                    student_logits=student_kk,
                                    teacher_logits=teacher_kk,
                                    labels=labels,
                                    kd_config_loss=loss_config
                                )
                                loss_value = (loss_value_vv + loss_value_kk) / 2
                            else:
                                self.logger.warning(
                                    f'{loss_key}: unknown kv_component "{kv_component}", '
                                    f'defaulting to "value"'
                                )
                                loss_value = loss_fn(
                                    student_logits=student_vv,
                                    teacher_logits=teacher_vv,
                                    labels=labels,
                                    kd_config_loss=loss_config
                                )
                            
                            layer_loss += float(loss_value)
                            num_valid_layers += 1
                        
                        if num_valid_layers == 0:
                            self.logger.warning(
                                f'{loss_key}: no valid layers found for past_key_values distillation, skipping'
                            )
                            continue
                        
                        # Average loss across selected layers
                        loss_value = layer_loss / num_valid_layers
                    else:
                        # If past_key_values is not a Cache, log warning
                        self.logger.warning(
                            f'{loss_key}: layer_index specified but past_key_values is not a Cache, '
                            f'skipping distillation'
                        )
                        continue
                else:
                    # No layer_index specified, compute loss for last layer only with relation matrices
                    if isinstance(student_past, tuple) and isinstance(teacher_past, tuple):
                        num_layers = min(len(student_past), len(teacher_past))
                        
                        # Get last layer index
                        layer_idx = num_layers - 1
                        
                        # Get key and value states for this layer
                        student_kv = student_past[layer_idx]
                        teacher_kv = teacher_past[layer_idx]
                        
                        # Extract key and value states
                        student_keys, student_values = student_kv[0], student_kv[1]
                        teacher_keys, teacher_values = teacher_kv[0], teacher_kv[1]
                        
                        # Compute V @ V^T for value-value relations
                        # Shape: (batch_size, num_kv_heads, seq_len, seq_len)
                        student_vv = torch.matmul(
                            student_values, student_values.transpose(-1, -2)
                        )
                        teacher_vv = torch.matmul(
                            teacher_values, teacher_values.transpose(-1, -2)
                        )
                        
                        # Scale and apply softmax for numerical stability and probability distribution
                        head_dim = student_values.shape[-1]
                        student_vv = student_vv / (head_dim ** 0.5)
                        teacher_vv = teacher_vv / (head_dim ** 0.5)
                        student_vv = F.softmax(student_vv, dim=-1)
                        teacher_vv = F.softmax(teacher_vv, dim=-1)
                        
                        # Handle head count mismatch
                        if student_vv.shape[1] != teacher_vv.shape[1]:
                            teacher_num_heads = teacher_vv.shape[1]
                            student_num_heads = student_vv.shape[1]
                            
                            if teacher_num_heads > student_num_heads:
                                heads_per_group = teacher_num_heads // student_num_heads
                                teacher_vv = teacher_vv.view(
                                    teacher_vv.shape[0],
                                    student_num_heads,
                                    heads_per_group,
                                    teacher_vv.shape[2],
                                    teacher_vv.shape[3]
                                ).mean(dim=2)
                            else:
                                self.logger.warning(
                                    f'{loss_key}: student has more kv heads than teacher, '
                                    f'skipping layer {layer_idx}'
                                )
                                continue
                        
                        # Compute loss for last layer
                        loss_value = loss_fn(
                            student_logits=student_vv,
                            teacher_logits=teacher_vv,
                            labels=labels,
                            kd_config_loss=loss_config
                        )
                    else:
                        self.logger.warning(
                            f'{loss_key}: past_key_values is not a tuple, skipping distillation'
                        )
                        continue
                
                weighted_loss = loss_config.alpha * float(loss_value)
                distill_loss += weighted_loss
                loss_dict['distill_loss_details'][loss_key] = float(loss_value)
            
            # Unknown loss type
            else:
                self.logger.warning(
                    f'{loss_key}: unknown loss_type "{loss_config.loss_type}", skipping'
                )
        
        # Finalize loss dictionary
        loss_dict['distill_loss'] = distill_loss
        
        # Compute total loss with task_loss alpha weighting
        # total_loss = loss_task_alpha * task_loss + distill_loss
        weighted_task_loss = self.config.loss_task_alpha * task_loss
        total_loss = weighted_task_loss + distill_loss
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def __repr__(self):
        """String representation of KD object."""
        num_losses = len(self.config.losses)
        loss_types = [cfg.loss_type for cfg in self.config.losses.values()]
        return (
            f'KD(method={self.config.method}, '
            f'num_losses={num_losses}, '
            f'loss_types={loss_types})'
        )
