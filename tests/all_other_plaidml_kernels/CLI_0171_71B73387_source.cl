#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i1 }
// Ranges: { 128 }
// Out stride: { 1 }
// Elementwise input X_I_32 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(Add)]] X_T96 = add(X_T63, X_I_32)
// Elementwise op: X_T97 = cmp_lt(X_T96, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T98 = cond(X_T97, X_T36, X_T96)
// Elementwise op: [[pid(Sqrt)]] X_T99 = sqrt(X_T98)
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
__kernel void kernel_c108_sdk_10(__global float* restrict  X_T99, __global const float* restrict  X_I_32)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 128);
  float LX_I_32 = X_I_32[i1_tid];
  float LX_T96 = (1.0009999641624745e-5f + LX_I_32);
  int LX_T97 = (LX_T96 < (float)0);
  float LX_T98 = select((float)LX_T96, (float)0, (int)LX_T97);
  float LX_T99 = native_sqrt(LX_T98);
  X_T99[i1_tid] = LX_T99;
}
