#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i1 }
// Ranges: { 128 }
// Out stride: { 1 }
// Elementwise input X_I_32 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(Add)]] X_T76 = add(X_T43, X_I_32)
// Elementwise op: X_T77 = cmp_lt(X_T76, X_T20)
// Elementwise op: [[pid(Sqrt)]] X_T78 = cond(X_T77, X_T20, X_T76)
// Elementwise op: [[pid(Sqrt)]] X_T79 = sqrt(X_T78)
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
__kernel void kernel_c68_sdk_10(__global float* restrict  X_T79, __global const float* restrict  X_I_32)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 128);
  float LX_I_32 = X_I_32[i1_tid];
  float LX_T76 = (1.0009999641624745e-5f + LX_I_32);
  int LX_T77 = (LX_T76 < (float)0);
  float LX_T78 = select((float)LX_T76, (float)0, (int)LX_T77);
  float LX_T79 = native_sqrt(LX_T78);
  X_T79[i1_tid] = LX_T79;
}
