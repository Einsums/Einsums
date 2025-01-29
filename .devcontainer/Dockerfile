FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-24.04

# This Dockerfile's base image has a non-root user with sudo access. Use the "remoteUser"
# property in devcontainer.json to use it. On Linux, the container's GID/UIDs will be
# updated to match your local UID/GID (when using the dockerfile property).
# See https://aka.ms/vscode-remote/containers/non-root-user for details.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
ARG TARGETARCH

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Configure apt and install packages
RUN apt-get update \
    #
    # Install C++ tools
    && apt-get -y install \
    build-essential \
    git \
    ninja-build \
    ccache \
    zsh \
    libblas-dev \
    liblapack-dev \
    libhdf5-serial-dev \
    valgrind \
    pybind11-dev \
    python3-numpy \
    python3-pybind11 \
    libfftw3-dev \
    catch2 \
    hdf5-tools \
    lsb-release \
    wget \
    software-properties-common \
    #
    # [Optional] Update UID/GID if needed
    && if [ "${USER_GID}" != "1000"] || [ "${USER_UID}" != "1000" ]; then \
    groupmod --gid ${USER_GID} ${USERNAME} \
    && usermod --uid ${USER_UID} --gid ${USER_GID} ${USERNAME} \
    && chown -R ${USER_UID}:${USER_GID} /home/${USERNAME}; \
    fi

RUN apt-get update \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
    && echo "deb https://apt.kitware.com/ubuntu/ noble main" >> /etc/apt/sources.list \
    && apt-get update \
    && apt-get -y install cmake

RUN wget https://apt.llvm.org/llvm.sh \
    && chmod +x llvm.sh \
    && ./llvm.sh 19 all

RUN if [ "$TARGETARCH" = "amd64" ]; then \
    curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB | gpg --dearmor | tee /usr/share/keyrings/intel-oneapi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main " > /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update \
    && apt-get upgrade -y \
    && \
    apt-get install -y --no-install-recommends \
      pkg-config gnupg libarchive13 wget net-tools intel-basekit intel-hpckit; \
    fi

# RUN add-apt-repository ppa:ubuntu-toolchain-r/test \
    # && apt update \
    # && apt install gcc-13 g++-13 -y \
    # && sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 13 --slave /usr/bin/g++ g++ /usr/bin/g++-13

# Cleanup the image
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Set links to solve ninja dependencies when compiling
RUN ln -s /usr/include /include

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog
ENV LANG=C.UTF-8

ENV LANG=C.UTF-8
ENV ADVISOR_2025_DIR='/opt/intel/oneapi/advisor/2025.0'
ENV APM='/opt/intel/oneapi/advisor/2025.0/perfmodels'
ENV CCL_CONFIGURATION='cpu_gpu_dpcpp'
ENV CCL_CONFIGURATION_PATH=''
ENV CCL_ROOT='/opt/intel/oneapi/ccl/2021.14'
ENV CLASSPATH='/opt/intel/oneapi/mpi/2021.14/share/java/mpi.jar'
ENV CMAKE_PREFIX_PATH='/opt/intel/oneapi/tbb/2022.0/env/..:/opt/intel/oneapi/pti/0.10/lib/cmake/pti:/opt/intel/oneapi/mkl/2025.0/lib/cmake:/opt/intel/oneapi/ipp/2022.0/lib/cmake/ipp:/opt/intel/oneapi/dpl/2022.7/lib/cmake/oneDPL:/opt/intel/oneapi/dnnl/2025.0/lib/cmake:/opt/intel/oneapi/dal/2025.0:/opt/intel/oneapi/compiler/2025.0'
ENV CMPLR_ROOT='/opt/intel/oneapi/compiler/2025.0'
ENV CPATH='/opt/intel/oneapi/umf/0.9/include:/opt/intel/oneapi/tbb/2022.0/env/../include:/opt/intel/oneapi/pti/0.10/include:/opt/intel/oneapi/mpi/2021.14/include:/opt/intel/oneapi/mkl/2025.0/include:/opt/intel/oneapi/ippcp/2025.0/include:/opt/intel/oneapi/ipp/2022.0/include:/opt/intel/oneapi/dpl/2022.7/include:/opt/intel/oneapi/dpcpp-ct/2025.0/include:/opt/intel/oneapi/dnnl/2025.0/include:/opt/intel/oneapi/dev-utilities/2025.0/include:/opt/intel/oneapi/dal/2025.0/include:/opt/intel/oneapi/ccl/2021.14/include'
ENV DALROOT='/opt/intel/oneapi/dal/2025.0'
ENV DAL_MAJOR_BINARY='3'
ENV DAL_MINOR_BINARY='0'
ENV DIAGUTIL_PATH='/opt/intel/oneapi/dpcpp-ct/2025.0/etc/dpct/sys_check/sys_check.sh:/opt/intel/oneapi/compiler/2025.0/etc/compiler/sys_check/sys_check.sh'
ENV DNNLROOT='/opt/intel/oneapi/dnnl/2025.0'
ENV DPL_ROOT='/opt/intel/oneapi/dpl/2022.7'
ENV FI_PROVIDER_PATH='/opt/intel/oneapi/mpi/2021.14/opt/mpi/libfabric/lib/prov:/usr/lib/x86_64-linux-gnu/libfabric'
ENV GDB_INFO='/opt/intel/oneapi/debugger/2025.0/share/info/'
ENV INFOPATH='/opt/intel/oneapi/debugger/2025.0/share/info'
ENV INTEL_PYTHONHOME='/opt/intel/oneapi/debugger/2025.0/opt/debugger'
ENV IPPCP_TARGET_ARCH='intel64'
ENV IPPCRYPTOROOT='/opt/intel/oneapi/ippcp/2025.0'
ENV IPPROOT='/opt/intel/oneapi/ipp/2022.0'
ENV IPP_TARGET_ARCH='intel64'
ENV I_MPI_ROOT='/opt/intel/oneapi/mpi/2021.14'
ENV LD_LIBRARY_PATH='/opt/intel/oneapi/tcm/1.2/lib:/opt/intel/oneapi/umf/0.9/lib:/opt/intel/oneapi/tbb/2022.0/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/pti/0.10/lib:/opt/intel/oneapi/mpi/2021.14/opt/mpi/libfabric/lib:/opt/intel/oneapi/mpi/2021.14/lib:/opt/intel/oneapi/mkl/2025.0/lib:/opt/intel/oneapi/ippcp/2025.0/lib/:/opt/intel/oneapi/ipp/2022.0/lib:/opt/intel/oneapi/dnnl/2025.0/lib:/opt/intel/oneapi/debugger/2025.0/opt/debugger/lib:/opt/intel/oneapi/dal/2025.0/lib:/opt/intel/oneapi/compiler/2025.0/opt/compiler/lib:/opt/intel/oneapi/compiler/2025.0/lib:/opt/intel/oneapi/ccl/2021.14/lib/'
ENV LIBRARY_PATH='/opt/intel/oneapi/tcm/1.2/lib:/opt/intel/oneapi/umf/0.9/lib:/opt/intel/oneapi/tbb/2022.0/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/pti/0.10/lib:/opt/intel/oneapi/mpi/2021.14/lib:/opt/intel/oneapi/mkl/2025.0/lib:/opt/intel/oneapi/ippcp/2025.0/lib/:/opt/intel/oneapi/ipp/2022.0/lib:/opt/intel/oneapi/dnnl/2025.0/lib:/opt/intel/oneapi/dal/2025.0/lib:/opt/intel/oneapi/compiler/2025.0/lib:/opt/intel/oneapi/ccl/2021.14/lib/'
ENV MANPATH='/opt/intel/oneapi/mpi/2021.14/share/man:/opt/intel/oneapi/debugger/2025.0/share/man:/opt/intel/oneapi/compiler/2025.0/share/man:'
ENV MKLROOT='/opt/intel/oneapi/mkl/2025.0'
ENV NLSPATH='/opt/intel/oneapi/compiler/2025.0/lib/compiler/locale/%l_%t/%N'
ENV OCL_ICD_FILENAMES='/opt/intel/oneapi/compiler/2025.0/lib/libintelocl.so'
ENV ONEAPI_ROOT='/opt/intel/oneapi'
ENV PATH='/opt/intel/oneapi/vtune/2025.0/bin64:/opt/intel/oneapi/mpi/2021.14/bin:/opt/intel/oneapi/mkl/2025.0/bin:/opt/intel/oneapi/dpcpp-ct/2025.0/bin:/opt/intel/oneapi/dev-utilities/2025.0/bin:/opt/intel/oneapi/debugger/2025.0/opt/debugger/bin:/opt/intel/oneapi/compiler/2025.0/bin:/opt/intel/oneapi/advisor/2025.0/bin64:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
ENV PKG_CONFIG_PATH='/opt/intel/oneapi/vtune/2025.0/include/pkgconfig/lib64:/opt/intel/oneapi/tbb/2022.0/env/../lib/pkgconfig:/opt/intel/oneapi/mpi/2021.14/lib/pkgconfig:/opt/intel/oneapi/mkl/2025.0/lib/pkgconfig:/opt/intel/oneapi/ippcp/2025.0/lib/pkgconfig:/opt/intel/oneapi/dpl/2022.7/lib/pkgconfig:/opt/intel/oneapi/dnnl/2025.0/lib/pkgconfig:/opt/intel/oneapi/dal/2025.0/lib/pkgconfig:/opt/intel/oneapi/compiler/2025.0/lib/pkgconfig:/opt/intel/oneapi/ccl/2021.14/lib/pkgconfig/:/opt/intel/oneapi/advisor/2025.0/include/pkgconfig/lib64:'
ENV PYTHONPATH='/opt/intel/oneapi/advisor/2025.0/pythonapi'
ENV Pti_DIR='/opt/intel/oneapi/pti/0.10/lib/cmake/pti'
ENV SETVARS_COMPLETED='1'
ENV TBBROOT='/opt/intel/oneapi/tbb/2022.0/env/..'
ENV TCM_ROOT='/opt/intel/oneapi/tcm/1.2'
ENV UMF_ROOT='/opt/intel/oneapi/umf/0.9'
ENV VTUNE_PROFILER_2025_DIR='/opt/intel/oneapi/vtune/2025.0'
ENV VTUNE_PROFILER_DIR='/opt/intel/oneapi/vtune/2025.0'
