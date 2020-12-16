#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i1 }
// Ranges: { 128 }
// Out stride: { 1 }
// Elementwise input X_I_32 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(Add)]] X_T104 = add(X_T71, X_I_32)
// Elementwise op: X_T105 = cmp_lt(X_T104, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T106 = cond(X_T105, X_T36, X_T104)
// Elementwise op: [[pid(Sqrt)]] X_T107 = sqrt(X_T106)
// Tile size: { 128 }
// Contraction output var shape: fp32(128):(1):512 bytes
// Computed true ops: 512
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 16
// Computed mem write: 512
// Computed operations: 128
// Computed rollups: 0
// Computed threads used: 128
// lwork = 128, 1, 1
// gwork = 128, 1, 1
__kernel void kernel_c124_sdk_10(__global float* restrict  X_T107, __global const float* restrict  X_I_32)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 128);
  float LX_I_32 = X_I_32[i1_tid];
  float LX_T104 = (1.0009999641624745e-5f + LX_I_32);
  int LX_T105 = (LX_T104 < (float)0);
  float LX_T106 = select((float)LX_T104, (float)0, (int)LX_T105);
  float LX_T107 = native_sqrt(LX_T106);
  X_T107[i1_tid] = LX_T107;
}
