#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 576 }
// Out stride: { 1 }
// Elementwise input X_I_313 shape: fp32(576):(1):2.25 KiB
// Elementwise op: [[pid(Add)]] X_T806 = add(X_T43, X_I_313)
// Elementwise op: X_T807 = cmp_lt(X_T806, X_T20)
// Elementwise op: [[pid(Sqrt)]] X_T808 = cond(X_T807, X_T20, X_T806)
// Elementwise op: [[pid(Sqrt)]] X_T809 = sqrt(X_T808)
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
__kernel void kernel_c68_sdk_271(__global float* restrict  X_T809, __global const float* restrict  X_I_313)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 512) || (i1_tid < 64));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_313 = X_I_313[gout_idx];
    float LX_T806 = (1.0009999641624745e-5f + LX_I_313);
    int LX_T807 = (LX_T806 < (float)0);
    float LX_T808 = select((float)LX_T806, (float)0, (int)LX_T807);
    float LX_T809 = native_sqrt(LX_T808);
    X_T809[gout_idx] = LX_T809;
  }
}
