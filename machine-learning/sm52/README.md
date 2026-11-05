# Immich ML for Maxwell GPUs (sm_52)

This image is the official `immich-machine-learning:<version>-cuda` image with one change. The `onnxruntime-gpu` wheel is rebuilt for sm_52 only. The GTX 950, 960, 970 and 980 need these kernels. The image has no kernels for Pascal or newer GPUs. Use the official image on those GPUs.

Background: Immich issue immich-app/immich#25732. Microsoft builds the stock wheel from ORT 1.21 for `60-real` and newer only.

## Why this build works

- ORT 1.27.1 still selects `52;60;70;75;80;86;89;90` as its default architectures when the CUDA compiler is older than 12.8 (`cmake/external/cuda_configuration.cmake`). Thus sm_52 is a supported configuration upstream. It is only absent from the published wheel.
- The build uses the CUDA 12.8 toolkit, the same toolkit as the upstream CUDA 12 wheel. The official image runs that wheel on its CUDA 12.2 runtime. CUDA 12.2 `nvcc` cannot compile the abseil version in ORT 1.27.1. CUDA 12.8 still compiles sm_52.
- ORT skips kernels that need a newer GPU, such as flash attention (SM80 and newer) and CUTLASS LLM kernels (SM75 and newer). These kernels do not compile for sm_52.
- The official image pins cuDNN `9.10.2.21`. The v2.4.1 image pins the same version, and it works on the GTX 970.
- Immich uses FP16 models only when `MACHINE_LEARNING_MODEL_REVISION` is not `main`. Keep the default. Maxwell has no fast FP16.

Verified on 2026-11-05 with `v3.3.6` and ORT `v1.27.1` on a GTX 970 (driver 580.178.04):

- `cuobjdump --list-elf` shows 203 kernels in `libonnxruntime_providers_cuda.so`. All are sm_52. There is no PTX.
- `test/test_cuda_ep.py` prints `CUDAExecutionProvider` and `run ok`.
- `test/predict_check.py` gets HTTP 200 for CLIP text, CLIP image, faces and OCR. Repeat calls take 10 to 40 ms. The log shows `['CUDAExecutionProvider', 'CPUExecutionProvider']` for each model.

## Build

The build does not need a GPU. Any x86_64 Linux host with Docker BuildKit or Podman can build it.

```bash
docker build \
  --build-arg IMMICH_VERSION=v3.3.6 \
  --build-arg ORT_VERSION=v1.27.1 \
  --build-arg PARALLEL=4 \
  -t immich-machine-learning:v3.3.6-cuda-sm52 .
```

With Podman, build with `--format docker`. The default OCI format drops the HEALTHCHECK of the official image.

On Windows with Podman, do not run the build from the Windows `podman` client. The client can lose its connection during a long step that prints no output. The build then continues in the VM without a log. Run `build.sh` in the Podman machine as a systemd unit, as shown in its header comment. It writes `build.log` in this folder. `push.sh` pushes the image to ghcr.io the same way.

To find the correct `ORT_VERSION` for a release, read the `onnxruntime-gpu` version in `machine-learning/uv.lock` at that tag. If the value is wrong, the last stage stops with an error message.

To update to a new Immich release, change `IMMICH_VERSION`. If that release locks the same ORT version, the build cache reuses the wheel and the build takes minutes. Only an ORT version change starts a full compile.

To keep the wheel apart from the build cache:

```bash
docker build --target wheel --output type=local,dest=./dist --build-arg ORT_VERSION=v1.27.1 .
```

## Resources

Measured on a Ryzen 7 3800X with a 16 GB, 12 CPU WSL2 VM and `PARALLEL=3`:

| Build | Wall time | Memory peak |
|---|---|---|
| Full build, including an image pull of about 20 min | 43 min | 9.6 GB |
| Rebuild with the same ORT version (ccache hits) | 5 min | 4.6 GB |

Allow about 30 to 40 GB of disk for the CUDA devel image, the ORT build tree and the build cache. This value is an estimate.

If the build stops with `Killed` or `signal 9`, the build ran out of memory. Decrease `PARALLEL`.

## Deploy

1. Push the image to a registry that the server can read, or build it on the server.
2. On the server, run the provider test. Expect `run ok`:
   ```bash
   docker run --rm --gpus all -v "$PWD/test:/s:ro" --entrypoint python <image> /s/test_cuda_ep.py /s/relu.onnx
   ```
3. In the compose file, change the `immich-machine-learning` image to the new tag. Keep the NVIDIA device reservation.
4. Restart the container. Run Face Detection and Smart Search from Administration > Jobs.
5. Make sure that the ML log has no `cudaErrorNoKernelImageForDevice`, and that `nvidia-smi` shows the ML process.

`test/predict_check.py` sends CLIP, face and OCR requests to a running ML container. Copy it into the container with `docker cp` and run it with `python`.

## Limits

- The build uses the CUDA 12 toolkit. CUDA 13 cannot compile for sm_52. When Immich moves the image to CUDA 13, keep the base at the last CUDA 12 release.
- NVIDIA driver branch 580 is the last branch that supports Maxwell.
- The build patches ORT to load execution provider libraries with `RTLD_LAZY`. ORT compiles its MoE kernels only for SM75 and newer, so a build for 52 alone has 19 undefined MoE functions. Immich models do not use MoE ops. If a model uses a MoE op, the ML worker stops with `symbol lookup error`. The fix for that case is to build the `contrib_ops/cuda/llm` sources for `75-real`.
- The image does not include TensorRT-RTX. Immich uses TensorRT-RTX only on Ampere and newer GPUs.
