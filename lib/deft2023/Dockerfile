# Build: sudo docker build -t <project_name> .
# Run: sudo docker run -v $(pwd):/workspace/project --gpus all -it --rm <project_name>


FROM nvidia/cuda:12.0.0-devel-ubuntu20.04


ENV CONDA_ENV_NAME=myenv
ENV PYTHON_VERSION=3.9


# Basic setup
RUN apt update
RUN apt install -y zsh \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true

# Set working directory
WORKDIR /workspace/project


# Install Miniconda and create main env
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda3.sh
RUN bash miniconda3.sh -b -p /conda \
    && echo export PATH=/conda/bin:$PATH >> .zshrc \
    && rm miniconda3.sh
ENV PATH="/conda/bin:${PATH}"
RUN conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION}


# Switch to zsh shell
SHELL ["/bin/zsh", "-c"]


# Install requirements
COPY pyproject.toml ./
RUN source activate ${CONDA_ENV_NAME} \
    && pip install poetry \
    && poetry install


# Set ${CONDA_ENV_NAME} to default virutal environment
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.zshrc
