ARG VARIANT
FROM buildpack-deps:${VARIANT}-curl

COPY einsums_env.yml /tmp/einsums_env.yml
COPY postCreateCommands.sh /tmp/postCreateCommands.sh

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# Local fix for issue with timezone data at 'buildpack-deps'. Ref# https://github.com/devcontainers/images/issues/1709
RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y reinstall --no-install-recommends tzdata

ARG INSTALL_INTEL
RUN if [ "${INSTALL_INTEL}" = "True" ]; then \
        wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
        && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list \
        && apt-get -y install intel-hpckit; \
    fi

# Cleanup the image
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
