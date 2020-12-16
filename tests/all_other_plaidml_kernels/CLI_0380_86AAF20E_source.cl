#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 672 }
// Out stride: { 1 }
// Elementwise input X_I_343 shape: fp32(672):(1):2.625 KiB
// Elementwise op: [[pid(Add)]] X_T909 = add(X_T71, X_I_343)
// Elementwise op: X_T910 = cmp_lt(X_T909, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T911 = cond(X_T910, X_T36, X_T909)
// Elementwise op: [[pid(Sqrt)]] X_T912 = sqrt(X_T911)
// Tile size: { 256 }
// Contraction output var shape: fp32(672):(1):2.625 KiB
// Computed true ops: 2688
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
__kernel void kernel_c124_sdk_298(__global float* restrict  X_T912, __global const float* restrict  X_I_343)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 512) || (i1_tid < 160));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_343 = X_I_343[gout_idx];
    float LX_T909 = (1.0009999641624745e-5f + LX_I_343);
    int LX_T910 = (LX_T909 < (float)0);
    float LX_T911 = select((float)LX_T909, (float)0, (int)LX_T910);
    float LX_T912 = native_sqrt(LX_T911);
    X_T912[gout_idx] = LX_T912;
  }
}
