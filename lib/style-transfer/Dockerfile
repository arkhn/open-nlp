# Use the NVIDIA CUDA image as the base
FROM nvidia/cuda:12.6.1-devel-ubi8

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.4.2 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv" \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PATH="/opt/poetry/bin:$PATH"

# Install system dependencies
RUN dnf update -y && dnf install -y \
    curl \
    gcc \
    gcc-c++ \
    make \
    python3.11 \
    python3.11-devel \
    python3-pip \
    && dnf clean all

# Set Python 3.11 as default
RUN alternatives --set python3 /usr/bin/python3.11

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set up project directory
RUN mkdir -p $PYSETUP_PATH
WORKDIR $PYSETUP_PATH

# Copy pyproject.toml
COPY pyproject.toml .

# Install project dependencies
RUN poetry install

# Set the default command
CMD ["python3"]
