#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 64 1 1
// lid: 64 1 1
// Names: { i1 }
// Ranges: { 64 }
// Out stride: { 1 }
// Elementwise input X_I_22 shape: fp32(64):(1):256 bytes
// Elementwise op: [[pid(Add)]] X_T72 = add(X_T71, X_I_22)
// Elementwise op: X_T73 = cmp_lt(X_T72, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T74 = cond(X_T73, X_T36, X_T72)
// Elementwise op: [[pid(Sqrt)]] X_T75 = sqrt(X_T74)
// Tile size: { 64 }
// Contraction output var shape: fp32(64):(1):256 bytes
// Computed true ops: 256
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 8
// Computed mem write: 256
// Computed operations: 64
// Computed rollups: 0
// Computed threads used: 64
// lwork = 64, 1, 1
// gwork = 64, 1, 1
__kernel void kernel_c124_sdk_2(__global float* restrict  X_T75, __global const float* restrict  X_I_22)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 64);
  float LX_I_22 = X_I_22[i1_tid];
  float LX_T72 = (1.0009999641624745e-5f + LX_I_22);
  int LX_T73 = (LX_T72 < (float)0);
  float LX_T74 = select((float)LX_T72, (float)0, (int)LX_T73);
  float LX_T75 = native_sqrt(LX_T74);
  X_T75[i1_tid] = LX_T75;
}
