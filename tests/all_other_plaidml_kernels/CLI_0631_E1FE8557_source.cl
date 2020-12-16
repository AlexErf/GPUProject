#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1568 }
// Out stride: { 1 }
// Elementwise input X_I_824 shape: fp32(1568):(1):6.125 KiB
// Elementwise op: [[pid(Add)]] X_T2121 = add(X_T63, X_I_824)
// Elementwise op: X_T2122 = cmp_lt(X_T2121, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T2123 = cond(X_T2122, X_T36, X_T2121)
// Elementwise op: [[pid(Sqrt)]] X_T2124 = sqrt(X_T2123)
// Tile size: { 256 }
// Contraction output var shape: fp32(1568):(1):6.125 KiB
// Computed true ops: 6272
// Computed work groups: 7
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 1, 1
__kernel void kernel_c108_sdk_736(__global float* restrict  X_T2124, __global const float* restrict  X_I_824)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1536) || (i1_tid < 32));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_824 = X_I_824[gout_idx];
    float LX_T2121 = (1.0009999641624745e-5f + LX_I_824);
    int LX_T2122 = (LX_T2121 < (float)0);
    float LX_T2123 = select((float)LX_T2121, (float)0, (int)LX_T2122);
    float LX_T2124 = native_sqrt(LX_T2123);
    X_T2124[gout_idx] = LX_T2124;
  }
}
