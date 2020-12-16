#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1536 }
// Out stride: { 1 }
// Elementwise input X_I_898 shape: fp32(1536):(1):6 KiB
// Elementwise op: [[pid(Add)]] X_T2495 = add(X_T33, X_I_898)
// Elementwise op: X_T2496 = cmp_lt(X_T2495, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T2497 = cond(X_T2496, X_T4, X_T2495)
// Elementwise op: [[pid(Sqrt)]] X_T2498 = sqrt(X_T2497)
// Tile size: { 256 }
// Contraction output var shape: fp32(1536):(1):6 KiB
// Computed true ops: 6144
// Computed work groups: 6
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 1, 1
__kernel void kernel_c51_sdk_816(__global float* restrict  X_T2498, __global const float* restrict  X_I_898)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int gout_idx = (i1_gid + i1_tid);
  float LX_I_898 = X_I_898[gout_idx];
  float LX_T2495 = (0.0010000000474974513f + LX_I_898);
  int LX_T2496 = (LX_T2495 < (float)0);
  float LX_T2497 = select((float)LX_T2495, (float)0, (int)LX_T2496);
  float LX_T2498 = native_sqrt(LX_T2497);
  X_T2498[gout_idx] = LX_T2498;
}
