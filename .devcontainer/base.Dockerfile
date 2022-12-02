FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-22.04

# This Dockerfile's base image has a non-root user with sudo access. Use the "remoteUser"
# property in devcontainer.json to use it. On Linux, the container's GID/UIDs will be
# updated to match your local UID/GID (when using the dockerfile property).
# See https://aka.ms/vscode-remote/containers/non-root-user for details.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

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
    # libblas-dev \
    # liblapack-dev \
    cmake \
    libhdf5-serial-dev \
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

# Obtain a newer version of CMake from kitware directly.
# RUN apt-get update \
#     && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
#     && echo "deb https://apt.kitware.com/ubuntu/ focal main" >> /etc/apt/sources.list \
#     && apt-get update \
#     && apt-get -y install cmake

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update \
    && apt install -y intel-hpckit

RUN wget https://apt.llvm.org/llvm.sh \
    && chmod +x llvm.sh \
    && ./llvm.sh 14 \
    && apt install -y libomp-14-dev

# Cleanup the image
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Set links to solve ninja dependencies when compiling
RUN ln -s /usr/include /include

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog
ENV LANG=C.UTF-8

ENV ACL_BOARD_VENDOR_PATH="/opt/Intel/OpenCLFPGA/oneAPI/Boards"
ENV ADVISOR_2022_DIR="/opt/intel/oneapi/advisor/2022.3.1"
ENV APM="/opt/intel/oneapi/advisor/2022.3.1/perfmodels"
ENV CCL_CONFIGURATION="cpu_gpu_dpcpp"
ENV CCL_ROOT="/opt/intel/oneapi/ccl/2021.7.1"
ENV CLASSPATH="/opt/intel/oneapi/mpi/2021.7.1//lib/mpi.jar:/opt/intel/oneapi/dal/2021.7.1/lib/onedal.jar"
ENV CLCK_ROOT="/opt/intel/oneapi/clck/2021.7.1"
ENV CMAKE_PREFIX_PATH="/opt/intel/oneapi/vpl/2022.2.5:/opt/intel/oneapi/tbb/2021.7.1/env/..:/opt/intel/oneapi/dnnl/2022.2.1/cpu_dpcpp_gpu_dpcpp/../lib/cmake:/opt/intel/oneapi/dal/2021.7.1:/opt/intel/oneapi/compiler/2022.2.1/linux/IntelDPCPP"
ENV CMPLR_ROOT="/opt/intel/oneapi/compiler/2022.2.1"
ENV CONDA_DEFAULT_ENV="intelpython-python3.9"
ENV CONDA_EXE="/opt/intel/oneapi/intelpython/latest/bin/conda"
ENV CONDA_PREFIX="/opt/intel/oneapi/intelpython/latest"
ENV CONDA_PROMPT_MODIFIER="(intelpython-python3.9) "
ENV CONDA_PYTHON_EXE="/opt/intel/oneapi/intelpython/latest/bin/python"
ENV CONDA_SHLVL="1"
ENV CPATH="/opt/intel/oneapi/vpl/2022.2.5/include:/opt/intel/oneapi/tbb/2021.7.1/env/../include:/opt/intel/oneapi/mpi/2021.7.1//include:/opt/intel/oneapi/mkl/2022.2.1/include:/opt/intel/oneapi/ippcp/2021.6.2/include:/opt/intel/oneapi/ipp/2021.6.2/include:/opt/intel/oneapi/dpl/2021.7.2/linux/include:/opt/intel/oneapi/dpcpp-ct/2022.2.1/include:/opt/intel/oneapi/dnnl/2022.2.1/cpu_dpcpp_gpu_dpcpp/include:/opt/intel/oneapi/dev-utilities/2021.7.1/include:/opt/intel/oneapi/dal/2021.7.1/include:/opt/intel/oneapi/ccl/2021.7.1/include/cpu_gpu_dpcpp"
ENV CPLUS_INCLUDE_PATH="/opt/intel/oneapi/clck/2021.7.1/include"
ENV DAALROOT="/opt/intel/oneapi/dal/2021.7.1"
ENV DALROOT="/opt/intel/oneapi/dal/2021.7.1"
ENV DAL_MAJOR_BINARY="1"
ENV DAL_MINOR_BINARY="1"
ENV DIAGUTIL_PATH="/opt/intel/oneapi/vtune/2022.4.1/sys_check/vtune_sys_check.py:/opt/intel/oneapi/dpcpp-ct/2022.2.1/sys_check/sys_check.sh:/opt/intel/oneapi/debugger/2021.7.1/sys_check/debugger_sys_check.py:/opt/intel/oneapi/compiler/2022.2.1/sys_check/sys_check.sh:/opt/intel/oneapi/advisor/2022.3.1/sys_check/advisor_sys_check.py:"
ENV DNNLROOT="/opt/intel/oneapi/dnnl/2022.2.1/cpu_dpcpp_gpu_dpcpp"
ENV DPCT_BUNDLE_ROOT="/opt/intel/oneapi/dpcpp-ct/2022.2.1"
ENV DPL_ROOT="/opt/intel/oneapi/dpl/2021.7.2"
ENV FI_PROVIDER_PATH="/opt/intel/oneapi/mpi/2021.7.1//libfabric/lib/prov:/usr/lib64/libfabric"
ENV FPGA_VARS_ARGS=""
ENV FPGA_VARS_DIR="/opt/intel/oneapi/compiler/2022.2.1/linux/lib/oclfpga"
ENV GDB_INFO="/opt/intel/oneapi/debugger/2021.7.1/documentation/info/"
ENV INFOPATH="/opt/intel/oneapi/debugger/2021.7.1/gdb/intel64/lib"
ENV INSPECTOR_2022_DIR="/opt/intel/oneapi/inspector/2022.3.1"
ENV INTELFPGAOCLSDKROOT="/opt/intel/oneapi/compiler/2022.2.1/linux/lib/oclfpga"
ENV INTEL_LICENSE_FILE="/opt/intel/licenses:/home/vscode/intel/licenses:/opt/intel/oneapi/clck/2021.7.1/licensing:/opt/intel/licenses:/home/vscode/intel/licenses:/Users/Shared/Library/Application Support/Intel/Licenses"
ENV INTEL_PYTHONHOME="/opt/intel/oneapi/debugger/2021.7.1/dep"
ENV IPPCP_TARGET_ARCH="intel64"
ENV IPPCRYPTOROOT="/opt/intel/oneapi/ippcp/2021.6.2"
ENV IPPROOT="/opt/intel/oneapi/ipp/2021.6.2"
ENV IPP_TARGET_ARCH="intel64"
ENV I_MPI_ROOT="/opt/intel/oneapi/mpi/2021.7.1"
ENV LD_LIBRARY_PATH="/opt/intel/oneapi/vpl/2022.2.5/lib:/opt/intel/oneapi/tbb/2021.7.1/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/mpi/2021.7.1//libfabric/lib:/opt/intel/oneapi/mpi/2021.7.1//lib/release:/opt/intel/oneapi/mpi/2021.7.1//lib:/opt/intel/oneapi/mkl/2022.2.1/lib/intel64:/opt/intel/oneapi/itac/2021.7.1/slib:/opt/intel/oneapi/ippcp/2021.6.2/lib/intel64:/opt/intel/oneapi/ipp/2021.6.2/lib/intel64:/opt/intel/oneapi/dnnl/2022.2.1/cpu_dpcpp_gpu_dpcpp/lib:/opt/intel/oneapi/debugger/2021.7.1/gdb/intel64/lib:/opt/intel/oneapi/debugger/2021.7.1/libipt/intel64/lib:/opt/intel/oneapi/debugger/2021.7.1/dep/lib:/opt/intel/oneapi/dal/2021.7.1/lib/intel64:/opt/intel/oneapi/compiler/2022.2.1/linux/lib:/opt/intel/oneapi/compiler/2022.2.1/linux/lib/x64:/opt/intel/oneapi/compiler/2022.2.1/linux/lib/oclfpga/host/linux64/lib:/opt/intel/oneapi/compiler/2022.2.1/linux/compiler/lib/intel64_lin:/opt/intel/oneapi/ccl/2021.7.1/lib/cpu_gpu_dpcpp"
ENV LIBRARY_PATH="/opt/intel/oneapi/vpl/2022.2.5/lib:/opt/intel/oneapi/tbb/2021.7.1/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/mpi/2021.7.1//libfabric/lib:/opt/intel/oneapi/mpi/2021.7.1//lib/release:/opt/intel/oneapi/mpi/2021.7.1//lib:/opt/intel/oneapi/mkl/2022.2.1/lib/intel64:/opt/intel/oneapi/ippcp/2021.6.2/lib/intel64:/opt/intel/oneapi/ipp/2021.6.2/lib/intel64:/opt/intel/oneapi/dnnl/2022.2.1/cpu_dpcpp_gpu_dpcpp/lib:/opt/intel/oneapi/dal/2021.7.1/lib/intel64:/opt/intel/oneapi/compiler/2022.2.1/linux/compiler/lib/intel64_lin:/opt/intel/oneapi/compiler/2022.2.1/linux/lib:/opt/intel/oneapi/clck/2021.7.1/lib/intel64:/opt/intel/oneapi/ccl/2021.7.1/lib/cpu_gpu_dpcpp"
ENV MANPATH="/opt/intel/oneapi/mpi/2021.7.1/man:/opt/intel/oneapi/itac/2021.7.1/man:/opt/intel/oneapi/debugger/2021.7.1/documentation/man:/opt/intel/oneapi/compiler/2022.2.1/documentation/en/man/common:/opt/intel/oneapi/clck/2021.7.1/man::"
ENV MKLROOT="/opt/intel/oneapi/mkl/2022.2.1"
ENV NLSPATH="/opt/intel/oneapi/mkl/2022.2.1/lib/intel64/locale/%l_%t/%N:/opt/intel/oneapi/compiler/2022.2.1/linux/compiler/lib/intel64_lin/locale/%l_%t/%N"
ENV OCL_ICD_FILENAMES="libintelocl_emu.so:libalteracl.so:/opt/intel/oneapi/compiler/2022.2.1/linux/lib/x64/libintelocl.so"
ENV ONEAPI_ROOT="/opt/intel/oneapi"
ENV PATH="/opt/intel/oneapi/vtune/2022.4.1/bin64:/opt/intel/oneapi/vpl/2022.2.5/bin:/opt/intel/oneapi/mpi/2021.7.1//libfabric/bin:/opt/intel/oneapi/mpi/2021.7.1//bin:/opt/intel/oneapi/mkl/2022.2.1/bin/intel64:/opt/intel/oneapi/itac/2021.7.1/bin:/opt/intel/oneapi/intelpython/latest/bin:/opt/intel/oneapi/intelpython/latest/condabin:/opt/intel/oneapi/inspector/2022.3.1/bin64:/opt/intel/oneapi/dpcpp-ct/2022.2.1/bin:/opt/intel/oneapi/dev-utilities/2021.7.1/bin:/opt/intel/oneapi/debugger/2021.7.1/gdb/intel64/bin:/opt/intel/oneapi/compiler/2022.2.1/linux/lib/oclfpga/bin:/opt/intel/oneapi/compiler/2022.2.1/linux/bin/intel64:/opt/intel/oneapi/compiler/2022.2.1/linux/bin:/opt/intel/oneapi/clck/2021.7.1/bin/intel64:/opt/intel/oneapi/advisor/2022.3.1/bin64:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/vscode/.local/bin"
ENV PKG_CONFIG_PATH="/opt/intel/oneapi/vtune/2022.4.1/include/pkgconfig/lib64:/opt/intel/oneapi/vpl/2022.2.5/lib/pkgconfig:/opt/intel/oneapi/tbb/2021.7.1/env/../lib/pkgconfig:/opt/intel/oneapi/mpi/2021.7.1/lib/pkgconfig:/opt/intel/oneapi/mkl/2022.2.1/lib/pkgconfig:/opt/intel/oneapi/ippcp/2021.6.2/lib/pkgconfig:/opt/intel/oneapi/inspector/2022.3.1/include/pkgconfig/lib64:/opt/intel/oneapi/dpl/2021.7.2/lib/pkgconfig:/opt/intel/oneapi/dnnl/2022.2.1/cpu_dpcpp_gpu_dpcpp/../lib/pkgconfig:/opt/intel/oneapi/dal/2021.7.1/lib/pkgconfig:/opt/intel/oneapi/compiler/2022.2.1/lib/pkgconfig:/opt/intel/oneapi/ccl/2021.7.1/lib/pkgconfig:/opt/intel/oneapi/advisor/2022.3.1/include/pkgconfig/lib64:"
ENV PYTHONPATH="/opt/intel/oneapi/advisor/2022.3.1/pythonapi"
ENV SETVARS_COMPLETED="1"
ENV TBBROOT="/opt/intel/oneapi/tbb/2021.7.1/env/.."
ENV VTUNE_PROFILER_2022_DIR="/opt/intel/oneapi/vtune/2022.4.1"
ENV VTUNE_PROFILER_DIR="/opt/intel/oneapi/vtune/2022.4.1"
ENV VT_ADD_LIBS="-ldwarf -lelf -lvtunwind -lm -lpthread"
ENV VT_LIB_DIR="/opt/intel/oneapi/itac/2021.7.1/lib"
ENV VT_MPI="impi4"
ENV VT_ROOT="/opt/intel/oneapi/itac/2021.7.1"
ENV VT_SLIB_DIR="/opt/intel/oneapi/itac/2021.7.1/slib"
ENV _CE_CONDA=""
ENV _CE_M=""

USER vscode