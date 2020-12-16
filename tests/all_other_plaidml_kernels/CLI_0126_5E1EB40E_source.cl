#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 576 }
// Out stride: { 1 }
// Elementwise input X_I_208 shape: fp32(576):(1):2.25 KiB
// Elementwise op: [[pid(Add)]] X_T483 = add(X_T59, X_I_208)
// Elementwise op: X_T484 = cmp_lt(X_T483, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T485 = cond(X_T484, X_T4, X_T483)
// Elementwise op: [[pid(Sqrt)]] X_T486 = sqrt(X_T485)
// Tile size: { 256 }
// Contraction output var shape: fp32(576):(1):2.25 KiB
// Computed true ops: 2304
// Computed work groups: 3
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 1, 1
__kernel void kernel_c43_sdk_129(__global float* restrict  X_T486, __global const float* restrict  X_I_208)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 512) || (i1_tid < 64));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_208 = X_I_208[gout_idx];
    float LX_T483 = (0.0010000000474974513f + LX_I_208);
    int LX_T484 = (LX_T483 < (float)0);
    float LX_T485 = select((float)LX_T483, (float)0, (int)LX_T484);
    float LX_T486 = native_sqrt(LX_T485);
    X_T486[gout_idx] = LX_T486;
  }
}
