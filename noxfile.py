"""Nox configuration for EdgeRazor multi-dependency testing.

Usage:
    nox                         # Run all sessions
    nox -s tests                # All torch × transformers combos
    nox -s tests-4.57.6-2.12.0 # Specific combo
    nox -s lint                 # Linting only
    nox -s e2e                  # End-to-end tests
    nox -l                      # List sessions
"""

import nox

TRANSFORMERS_VERSIONS = ["4.55.0", "4.56.0", "4.57.6"]
TORCH_VERSIONS = ["2.8.0", "2.9.0", "2.10.0", "2.11.0", "2.12.0"]

# torchvision versions paired with torch releases
TORCHVISION_MAP = {
    "2.8.0": "0.23.0",
    "2.9.0": "0.24.0",
    "2.10.0": "0.25.0",
    "2.11.0": "0.26.0",
    "2.12.0": "0.27.0",
}


def _install_deps(session, transformers_version, torch_version):
    """Install test dependencies for a specific transformers + torch combo."""
    session.install("pytest>=7.0", "pytest-cov>=4.0", "pytest-mock>=3.10")
    session.install("pyyaml==6.0.3", "pyfiglet>=1.0.0", "colorama>=0.4.0")
    session.install(f"transformers=={transformers_version}")

    tv = TORCHVISION_MAP.get(torch_version, "0.23.0")
    session.install(f"torch=={torch_version}", f"torchvision=={tv}")

    session.install("-e", ".[dev]", silent=False)


@nox.session
@nox.parametrize("tf", TRANSFORMERS_VERSIONS)
@nox.parametrize("torch_v", TORCH_VERSIONS)
def tests(session, tf, torch_v):
    """Run unit and integration tests for a specific combo."""
    _install_deps(session, tf, torch_v)

    session.run("python", "-c", "import torch; print(f'torch: {torch.__version__}')")
    session.run("python", "-c", "import transformers; print(f'transformers: {transformers.__version__}')")

    session.run(
        "pytest", "tests/unit", "tests/integration",
        "-v", "--tb=short",
        *session.posargs,
    )


@nox.session
def e2e(session):
    """Run end-to-end tests."""
    _install_deps(session, "4.57.6", "2.12.0")
    session.run("pytest", "tests/e2e", "-v", "--tb=short", *session.posargs)


@nox.session
def lint(session):
    """Run ruff linter."""
    session.install("ruff>=0.13.1")
    session.run("ruff", "check", "src/edgerazor")
    session.run("ruff", "check", "tests/")


@nox.session
def format_check(session):
    """Check code formatting with ruff."""
    session.install("ruff>=0.13.1")
    session.run("ruff", "format", "--check", "src/edgerazor")
    session.run("ruff", "format", "--check", "tests/")


@nox.session
def coverage(session):
    """Run full test suite and generate HTML coverage report."""
    _install_deps(session, "4.57.6", "2.12.0")
    session.run(
        "pytest", "tests/",
        "--cov=edgerazor",
        "--cov-report=html",
        "--cov-report=term",
        *session.posargs,
    )
    session.log("Coverage report generated at: htmlcov/index.html")
