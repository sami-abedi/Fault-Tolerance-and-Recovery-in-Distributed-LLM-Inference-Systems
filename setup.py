"""Package setup for CrashSafe."""
from setuptools import setup, find_packages

setup(
    name="crashsafe",
    version="0.1.0",
    description="Fault Tolerance and Recovery in Distributed LLM Inference Systems",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.30.0",
        "pydantic>=2.7.0",
        "pydantic-settings>=2.3.0",
        "httpx>=0.27.0",
        "pyyaml>=6.0.1",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "structlog>=24.2.0",
        "tenacity>=8.3.0",
        "psutil>=5.9.0",
    ],
)
