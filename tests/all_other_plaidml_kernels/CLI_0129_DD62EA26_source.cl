#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 384 }
// Out stride: { 1 }
// Elementwise input X_I_323 shape: fp32(384):(1):1.5 KiB
// Elementwise op: [[pid(Add)]] X_T889 = add(X_T33, X_I_323)
// Elementwise op: X_T890 = cmp_lt(X_T889, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T891 = cond(X_T890, X_T4, X_T889)
// Elementwise op: [[pid(Sqrt)]] X_T892 = sqrt(X_T891)
// Tile size: { 256 }
// Contraction output var shape: fp32(384):(1):1.5 KiB
// Computed true ops: 1536
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
__kernel void kernel_c51_sdk_289(__global float* restrict  X_T892, __global const float* restrict  X_I_323)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 256) || (i1_tid < 128));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_323 = X_I_323[gout_idx];
    float LX_T889 = (0.0010000000474974513f + LX_I_323);
    int LX_T890 = (LX_T889 < (float)0);
    float LX_T891 = select((float)LX_T889, (float)0, (int)LX_T890);
    float LX_T892 = native_sqrt(LX_T891);
    X_T892[gout_idx] = LX_T892;
  }
}
