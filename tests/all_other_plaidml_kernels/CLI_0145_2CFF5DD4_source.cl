#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i1 }
// Ranges: { 128 }
// Out stride: { 1 }
// Elementwise input X_I_351 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(Add)]] X_T952 = add(X_T33, X_I_351)
// Elementwise op: X_T953 = cmp_lt(X_T952, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T954 = cond(X_T953, X_T4, X_T952)
// Elementwise op: [[pid(Sqrt)]] X_T955 = sqrt(X_T954)
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
__kernel void kernel_c51_sdk_311(__global float* restrict  X_T955, __global const float* restrict  X_I_351)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 128);
  float LX_I_351 = X_I_351[i1_tid];
  float LX_T952 = (0.0010000000474974513f + LX_I_351);
  int LX_T953 = (LX_T952 < (float)0);
  float LX_T954 = select((float)LX_T952, (float)0, (int)LX_T953);
  float LX_T955 = native_sqrt(LX_T954);
  X_T955[i1_tid] = LX_T955;
}
