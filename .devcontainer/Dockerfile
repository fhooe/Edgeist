# stage to build the required OpenOCD fork.
FROM debian:trixie AS build

# hadolint ignore=DL3003,DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    libtool \
    automake \
    make \
    pkg-config \
    gcc-12 \
    libusb-1.0-0-dev && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 1 && \
    git clone https://github.com/STMicroelectronics/OpenOCD.git && \
    cd OpenOCD && \
    ./bootstrap && \
    ./configure --enable-stlink && \
    make && make install

# build the devcontainer
FROM debian:trixie

ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# install OpenOCD
COPY --from=build /usr/local/bin/openocd /usr/local/bin/openocd
COPY --from=build /usr/local/share/openocd /usr/local/share/openocd

# add default non root user (with sudo permissions) and install required packages
# hadolint ignore=SC1091,DL3008
RUN --mount=type=bind,source=requirements.txt,target=/requirements.txt apt-get update && \
    apt-get install -y --no-install-recommends \
    sudo \
    ca-certificates \
    bash-completion \
    curl \
    build-essential \
    cmake \
    git \
    ssh \
    locales \
    vim \
    less \
    python3 \
    python3-venv \
    python3-pip \
    shellcheck \
    clangd \
    clang-tidy \
    cmake-format \
    ninja-build \
    gcc-arm-none-eabi \
    libnewlib-arm-none-eabi \
    libusb-1.0-0 \
    gdb-multiarch \
    codespell && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    # Install hadolint
    curl -fsSL https://github.com/hadolint/hadolint/releases/download/v2.12.0/hadolint-Linux-x86_64 -o /usr/local/bin/hadolint && \
    chmod 755 /usr/local/bin/hadolint && \
    # Add default non root user (with root permissions)
    groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    # install english locale
    sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen && \
    bash -c 'echo "#!/bin/bash" > /usr/local/bin/sudoopenocd' && \
    bash -c 'echo "sudo openocd \"\$@\"" >> /usr/local/bin/sudoopenocd' && \
    chmod +x /usr/local/bin/sudoopenocd && \
    # install python dependencies
    python3 -m venv /venv && \
    . /venv/bin/activate && \
    pip install --no-cache-dir -r /requirements.txt

# *activate* venv
ENV PATH="/venv/bin:$PATH"

# set locales
ENV LANG=en_US.UTF-8 
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# set default user
USER $USERNAME

RUN bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh)"

# expose required ports
EXPOSE 6666
EXPOSE 4444
EXPOSE 3333
