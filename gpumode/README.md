# Triton Wheel Build Info

- **Wheel**: triton-3.7.0+gitb7fa781f-cp313-cp313-linux_x86_64.whl
- **Built**: 2026-05-06
- **Python**: CPython 3.13.13 (uv managed)
- **Platform**: linux-x86_64

## Source
- **Triton repo**: https://github.com/triton-lang/triton.git
- **Triton commit**: b7fa781f9 (`release/3.7.x` HEAD, `Split RemoveLayoutConversions cleanup so scf.if non-convergence is non fatal (#10174)`)
- **LLVM commit**: ac5dc54d5091 (from `cmake/llvm-hash.txt`)

## Patches (on top of b7fa781f9)

1. **`python/src/ir.cc`**: Fix plugin op builder to support return values — inserts a placeholder `Value()` at `args[0]` before calling `op.addOp`, then returns `args[0]` as the result. This is required for uTLX plugin ops that produce output values.

```diff
--- a/python/src/ir.cc
+++ b/python/src/ir.cc
@@ -1872,7 +1872,9 @@
       TritonOpBuilderBinding.def(
           op.name, [op](TritonOpBuilder &self, std::vector<Value> args) {
+            args.insert(args.begin(), Value());
             op.addOp(self, args);
+            return args[0];
           });
```

**Build flag**: Must build with `-DTRITON_EXT_ENABLED=ON` to enable plugin loading.

## Build Environment
- **CUDA ptxas**: 12.8.93 (Hopper), 13.1.80 (Blackwell)
- **CUDA CRT/RT**: 13.1
- **CUPTI**: 12.8.90
- **LLVM**: built from source with `mlir;lld`, targets `host;NVPTX;AMDGPU`
- **Backends**: nvidia, amd
- **Unit tests**: OFF
- **Proton**: OFF

## Build Script

```bash
# 1. Install Python 3.13
uv python install 3.13
uv venv --python 3.13 /home/wychi/oss/triton/.venv313
source /home/wychi/oss/triton/.venv313/bin/activate
uv pip install build cmake ninja setuptools wheel pybind11

# 2. Build LLVM (commit must match triton/cmake/llvm-hash.txt)
cd /home/wychi/oss/llvm-project
git checkout ac5dc54d509169d387fcfd495d71853d81c46484
cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir;lld" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
  -DLLVM_ENABLE_ASSERTIONS=ON
cmake --build build -j$(nproc)

# 3. Build Triton wheel
cd /home/wychi/oss/triton
export LLVM_SYSPATH=/home/wychi/oss/llvm-project/build
export TRITON_PTXAS_PATH=/usr/local/cuda-12.8/bin/ptxas
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda-13.1/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda-13.1/bin/cuobjdump
export TRITON_NVDISASM_PATH=/usr/local/cuda-13.1/bin/nvdisasm
export TRITON_CUDACRT_PATH=/usr/local/cuda-13.1/
export TRITON_CUDART_PATH=/usr/local/cuda-13.1/
export TRITON_CUPTI_PATH=/usr/local/cuda-12.8/
export TRITON_BUILD_UT=OFF
export TRITON_BUILD_PROTON=OFF
export TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF -DTRITON_EXT_ENABLED=ON"
python -m build --wheel --no-isolation

# 4. Output
ls dist/*.whl
```
