#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 320 }
// Out stride: { 1 }
// Elementwise input X_I_261 shape: fp32(320):(1):1.25 KiB
// Elementwise op: [[pid(Add)]] X_T699 = add(X_T59, X_I_261)
// Elementwise op: X_T700 = cmp_lt(X_T699, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T701 = cond(X_T700, X_T4, X_T699)
// Elementwise op: [[pid(Sqrt)]] X_T702 = sqrt(X_T701)
// Tile size: { 256 }
// Contraction output var shape: fp32(320):(1):1.25 KiB
// Computed true ops: 1280
// Computed work groups: 2
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 512, 1, 1
__kernel void kernel_c43_sdk_190(__global float* restrict  X_T702, __global const float* restrict  X_I_261)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 256) || (i1_tid < 64));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_261 = X_I_261[gout_idx];
    float LX_T699 = (0.0010000000474974513f + LX_I_261);
    int LX_T700 = (LX_T699 < (float)0);
    float LX_T701 = select((float)LX_T699, (float)0, (int)LX_T700);
    float LX_T702 = native_sqrt(LX_T701);
    X_T702[gout_idx] = LX_T702;
  }
}
