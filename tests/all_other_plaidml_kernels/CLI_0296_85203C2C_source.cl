#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 672 }
// Out stride: { 1 }
// Elementwise input X_I_343 shape: fp32(672):(1):2.625 KiB
// Elementwise op: [[pid(Add)]] X_T881 = add(X_T43, X_I_343)
// Elementwise op: X_T882 = cmp_lt(X_T881, X_T20)
// Elementwise op: [[pid(Sqrt)]] X_T883 = cond(X_T882, X_T20, X_T881)
// Elementwise op: [[pid(Sqrt)]] X_T884 = sqrt(X_T883)
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
__kernel void kernel_c68_sdk_298(__global float* restrict  X_T884, __global const float* restrict  X_I_343)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 512) || (i1_tid < 160));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_343 = X_I_343[gout_idx];
    float LX_T881 = (1.0009999641624745e-5f + LX_I_343);
    int LX_T882 = (LX_T881 < (float)0);
    float LX_T883 = select((float)LX_T881, (float)0, (int)LX_T882);
    float LX_T884 = native_sqrt(LX_T883);
    X_T884[gout_idx] = LX_T884;
  }
}
