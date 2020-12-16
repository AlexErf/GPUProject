#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2048 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 2048 }
// Out stride: { 1 }
// Elementwise input X_I_308 shape: fp32(2048):(1):8 KiB
// Elementwise op: [[pid(Add)]] X_T588 = add(X_T37, X_I_308)
// Elementwise op: X_T589 = cmp_lt(X_T588, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T590 = cond(X_T589, X_T3, X_T588)
// Elementwise op: [[pid(Sqrt)]] X_T591 = sqrt(X_T590)
// Tile size: { 256 }
// Contraction output var shape: fp32(2048):(1):8 KiB
// Computed true ops: 8192
// Computed work groups: 8
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2048, 1, 1
__kernel void kernel_c29_sdk_139(__global float* restrict  X_T591, __global const float* restrict  X_I_308)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int gout_idx = (i1_gid + i1_tid);
  float LX_I_308 = X_I_308[gout_idx];
  float LX_T588 = (0.0010000000474974513f + LX_I_308);
  int LX_T589 = (LX_T588 < (float)0);
  float LX_T590 = select((float)LX_T588, (float)0, (int)LX_T589);
  float LX_T591 = native_sqrt(LX_T590);
  X_T591[gout_idx] = LX_T591;
}
