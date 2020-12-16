#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 608 }
// Out stride: { 1 }
// Elementwise input X_I_323 shape: fp32(608):(1):2.375 KiB
// Elementwise op: [[pid(Add)]] X_T831 = add(X_T43, X_I_323)
// Elementwise op: X_T832 = cmp_lt(X_T831, X_T20)
// Elementwise op: [[pid(Sqrt)]] X_T833 = cond(X_T832, X_T20, X_T831)
// Elementwise op: [[pid(Sqrt)]] X_T834 = sqrt(X_T833)
// Tile size: { 128 }
// Contraction output var shape: fp32(608):(1):2.375 KiB
// Computed true ops: 2432
// Computed work groups: 5
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 16
// Computed mem write: 512
// Computed operations: 128
// Computed rollups: 0
// Computed threads used: 128
// lwork = 256, 1, 1
// gwork = 1280, 1, 1
__kernel void kernel_c68_sdk_280(__global float* restrict  X_T834, __global const float* restrict  X_I_323)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 128);
  int i1_tid = (tid % 128);
  int i1_cond = ((i1_gid != 512) || (i1_tid < 96));
  if (i1_cond)
  {
    if ((tid < 128))
    {
      int gout_idx = (i1_gid + i1_tid);
      float LX_I_323 = X_I_323[gout_idx];
      float LX_T831 = (1.0009999641624745e-5f + LX_I_323);
      int LX_T832 = (LX_T831 < (float)0);
      float LX_T833 = select((float)LX_T831, (float)0, (int)LX_T832);
      float LX_T834 = native_sqrt(LX_T833);
      X_T834[gout_idx] = LX_T834;
    }
  }
}
