#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 192 }
// Out stride: { 1 }
// Elementwise input X_I_71 shape: fp32(192):(1):768 bytes
// Elementwise op: [[pid(Add)]] X_T185 = add(X_T63, X_I_71)
// Elementwise op: X_T186 = cmp_lt(X_T185, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T187 = cond(X_T186, X_T36, X_T185)
// Elementwise op: [[pid(Sqrt)]] X_T188 = sqrt(X_T187)
// Tile size: { 192 }
// Contraction output var shape: fp32(192):(1):768 bytes
// Computed true ops: 768
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 24
// Computed mem write: 768
// Computed operations: 192
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 256, 1, 1
__kernel void kernel_c108_sdk_43(__global float* restrict  X_T188, __global const float* restrict  X_I_71)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  int i1_cond = (i1_tid < 192);
  if (i1_cond)
  {
    float LX_I_71 = X_I_71[i1_tid];
    float LX_T185 = (1.0009999641624745e-5f + LX_I_71);
    int LX_T186 = (LX_T185 < (float)0);
    float LX_T187 = select((float)LX_T185, (float)0, (int)LX_T186);
    float LX_T188 = native_sqrt(LX_T187);
    X_T188[i1_tid] = LX_T188;
  }
}
