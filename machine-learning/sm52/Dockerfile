# Immich machine-learning CUDA image with onnxruntime-gpu rebuilt for Maxwell (sm_52).
#
# Stage 1 builds the onnxruntime-gpu wheel. It depends only on ORT_VERSION, so the
# build cache reuses it for every Immich release that ships the same ORT version.
# Stage 2 takes the official image and replaces its onnxruntime-gpu wheel.
#
#   docker build --build-arg IMMICH_VERSION=v3.3.6 --build-arg ORT_VERSION=v1.27.1 \
#     --build-arg PARALLEL=4 -t immich-machine-learning:v3.3.6-cuda-sm52 .

ARG IMMICH_VERSION=v3.3.6

# Same Python that the official cuda image copies in (see builder-cuda upstream).
FROM python:3.13-slim-bookworm@sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26 AS python

# Upstream builds the CUDA 12 wheel with CUDA 12.8. The official image runs that wheel on its 12.2 runtime.
# nvcc 12.2 cannot compile the abseil version in ORT 1.27.1. CUDA 13 cannot compile sm_52.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS ort-build

ARG ORT_VERSION=v1.27.1
# Parallel compile jobs. Each job can use several GB of RAM. See README.md.
ARG PARALLEL=4
ENV CCACHE_DIR=/ccache DEBIAN_FRONTEND=noninteractive

# cuDNN pin matches the runtime pin in the official image.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ccache \
      libcudnn9-cuda-12=9.10.2.21-1 libcudnn9-headers-cuda-12=9.10.2.21-1 libcudnn9-dev-cuda-12=9.10.2.21-1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=python /usr/local/bin/ /usr/local/bin/
COPY --from=python /usr/local/lib/ /usr/local/lib/
COPY --from=python /usr/local/include/ /usr/local/include/
RUN ldconfig && python3 -m pip install --no-cache-dir "cmake>=3.28" ninja numpy packaging setuptools wheel

RUN git clone --depth 1 --branch "${ORT_VERSION}" https://github.com/microsoft/onnxruntime /code/onnxruntime
WORKDIR /code/onnxruntime

# ponytail: ORT compiles the MoE (contrib_ops/cuda/llm) kernels only for SM75 and newer. A build for 52 alone
# leaves those functions undefined, and RTLD_NOW then fails the CUDA provider load. RTLD_LAZY binds a function
# at its first call. Immich models do not use MoE ops. A model with a MoE op stops the worker with
# "symbol lookup error". Upgrade path: build the llm sources for 75-real in cmake/onnxruntime_providers_cuda.cmake.
RUN sed -i 's/dlopen(library_filename.c_str(), RTLD_NOW |/dlopen(library_filename.c_str(), RTLD_LAZY |/' onnxruntime/core/platform/posix/env.cc && \
    grep -q 'RTLD_LAZY | (global_symbols' onnxruntime/core/platform/posix/env.cc

RUN --mount=type=cache,target=/ccache \
    python3 tools/ci_build/build.py \
    --allow_running_as_root \
    --build_dir build \
    --config Release \
    --update \
    --build \
    --build_wheel \
    --parallel "${PARALLEL}" \
    --nvcc_threads 1 \
    --cmake_generator Ninja \
    --use_cuda \
    --cuda_version 12.8 \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr \
    --skip_tests \
    --use_cache \
    --compile_no_warning_as_error \
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=52 onnxruntime_BUILD_UNIT_TESTS=OFF onnxruntime_USE_FPA_INTB_GEMM=OFF && \
    mkdir /dist && cp build/Release/dist/onnxruntime_gpu-*.whl /dist/

# `docker build --target wheel --output type=local,dest=./dist .` exports only the wheel.
FROM scratch AS wheel
COPY --from=ort-build /dist/ /

FROM ghcr.io/immich-app/immich-machine-learning:${IMMICH_VERSION}-cuda

ARG ORT_VERSION=v1.27.1

# Fail when this Immich release locks a different ORT version than the wheel.
# Remove the TensorRT-RTX plugin. It does not support Maxwell. When Immich finds it, Immich drops the CUDA
# provider (_providers_default in immich_ml/sessions/ort.py). Without the plugin, Immich always uses CUDA.
RUN --mount=type=bind,from=wheel,target=/dist \
    --mount=type=bind,from=ghcr.io/astral-sh/uv:0.8.15@sha256:a5727064a0de127bdb7c9d3c1383f3a9ac307d9f2d8a391edc7896c54289ced0,source=/uv,target=/bin/uv \
    locked="$(python -c 'import importlib.metadata as m; print(m.version("onnxruntime-gpu"))')" && \
    if [ "$locked" != "${ORT_VERSION#v}" ]; then \
      echo "Image has onnxruntime-gpu $locked but ORT_VERSION is ${ORT_VERSION}. Set ORT_VERSION=v$locked." >&2; exit 1; \
    fi && \
    uv pip install --no-cache --python /opt/venv/bin/python --no-deps --reinstall /dist/onnxruntime_gpu-*.whl && \
    (uv pip uninstall --no-cache --python /opt/venv/bin/python onnxruntime-ep-nv-tensorrt-rtx-cu12 || true)
