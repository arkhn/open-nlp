# Cuda-based image which installs a Python Poetry environment designed to be used interactively.
#
# To build this image, use: `docker build -t <image-name> --build-arg PROJECT_NAME=<project-name> .`
# To use this image interactively, use:
# `docker run -v $(pwd):/opt/pysetup/ --gpus all -it --rm <image-name>`
# You might need to add `sudo` at the beginning of the commands.
# 
# To customize this image, you can use this example:
# docker build -t <image-name> --build-arg PROJECT_NAME=<project-name> . \
#   --build-arg ALIAS_USERNAME=<username> \
#   --build-arg ALIAS_REPOSITORY=dotfiles \
#   --build-arg ALIAS_PATH=.config/oh-my-zsh \
#   --build-arg NEOVIM_USERNAME=<username> \
#   --build-arg NEOVIM_DEPS_INSTALL_COMMAND="apt install -y npm zip" \
#   --build-arg NEOVIM_TOOLS_INSTALL_COMMAND="+MasonInstallAll"
# If this has been run with another configuration, you might need to destroy the cache with 
# `--no-cache` to introduce a new configuration.

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG PROJECT_NAME

ARG PYTHON_VERSION=3.11
ARG POETRY_VERSION=1.6.1
ARG ALIAS_USERNAME=ohmyzsh
ARG ALIAS_REPOSITORY=ohmyzsh
ARG ALIAS_PATH=custom
ARG NEOVIM_USERNAME=nvim-lua
ARG NEOVIM_REPOSITORY=kickstart.nvim
ARG NEOVIM_DEPS_INSTALL_COMMAND=""
ARG NEOVIM_PLUGIN_INSTALL_COMMAND="'+Lazy install'"
ARG NEOVIM_TOOLS_INSTALL_COMMAND=""

    # python
ENV PYTHON_VERSION=$PYTHON_VERSION \
    \
    # poetry
    # https://python-poetry.org/docs/configuration/#using-environment-variables
    POETRY_VERSION=$POETRY_VERSION \
    # make poetry install to this location
    POETRY_HOME="/opt/poetry" \
    # make poetry NOT create the virtual environment in the project's root, as it would be 
    # overwriten when setting a volume on /opt/pysetup/
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    \
    # paths
    CONFIG_PATH="/root/.config" \
    SETUP_PATH="/opt/setup" \
    PYSETUP_PATH="/opt/pysetup" 

# prepend poetry to path
ENV PATH="$POETRY_HOME/bin:$PATH"

# core dependencies
# setup timezone for software-properties-common install
ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# install dependencies
# software-properties-common is needed to run add-apt-repository
RUN apt update
RUN apt install -y \
    build-essential \
    ca-certificates \
    curl \
    git \
    software-properties-common \
    zsh \
    && rm -rf /var/lib/apt/lists

# install python
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install -y \
    python${PYTHON_VERSION} \
    python3-distutils \
    python3-pip \
    python3-venv

# install poetry via the official installer:
# https://python-poetry.org/docs/master/#installing-with-the-official-installer
# this script respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSL https://install.python-poetry.org | python3 -

# install oh-my-zsh & custom aliases (optional)
RUN curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh | zsh
RUN git clone https://github.com/$ALIAS_USERNAME/$ALIAS_REPOSITORY $SETUP_PATH/$ALIAS_REPOSITORY
RUN cp -r $SETUP_PATH/$ALIAS_REPOSITORY/$ALIAS_PATH/* $HOME/.oh-my-zsh/custom

# install neovim (optional)
RUN add-apt-repository ppa:neovim-ppa/unstable
RUN apt update
RUN apt install -y neovim

# install neovim custom config (optional)
RUN git clone https://github.com/$NEOVIM_USERNAME/$NEOVIM_REPOSITORY $CONFIG_PATH/nvim
# dependencies required for external tools
RUN $NEOVIM_DEPS_INSTALL_COMMAND
# install neovim plugins
RUN nvim $NEOVIM_PLUGIN_INSTALL_COMMAND +qall
# install neovim external tools
RUN nvim $NEOVIM_TOOLS_INSTALL_COMMAND +qall

# copy project requirement files here to ensure they will be cached.
WORKDIR $PYSETUP_PATH
COPY poetry.lock pyproject.toml ./

# install runtime deps
RUN poetry env use $PYTHON_VERSION 
RUN poetry install --only main --no-root

# copy local code
COPY $PROJECT_NAME ./$PROJECT_NAME
COPY README.md ./

# quicker install as runtime deps are already installed
RUN poetry install --only main
 
# use a zsh shell when started interactively
CMD ["/bin/zsh"]
