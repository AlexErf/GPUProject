#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3840 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 928 }
// Out stride: { 1 }
// Elementwise input X_I_423 shape: fp32(928):(1):3.625 KiB
// Elementwise op: [[pid(Add)]] X_T1109 = add(X_T71, X_I_423)
// Elementwise op: X_T1110 = cmp_lt(X_T1109, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1111 = cond(X_T1110, X_T36, X_T1109)
// Elementwise op: [[pid(Sqrt)]] X_T1112 = sqrt(X_T1111)
// Tile size: { 64 }
// Contraction output var shape: fp32(928):(1):3.625 KiB
// Computed true ops: 3712
// Computed work groups: 15
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 8
// Computed mem write: 256
// Computed operations: 64
// Computed rollups: 0
// Computed threads used: 64
// lwork = 256, 1, 1
// gwork = 3840, 1, 1
__kernel void kernel_c124_sdk_370(__global float* restrict  X_T1112, __global const float* restrict  X_I_423)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 64);
  int i1_tid = (tid % 64);
  int i1_cond = ((i1_gid != 896) || (i1_tid < 32));
  if (i1_cond)
  {
    if ((tid < 64))
    {
      int gout_idx = (i1_gid + i1_tid);
      float LX_I_423 = X_I_423[gout_idx];
      float LX_T1109 = (1.0009999641624745e-5f + LX_I_423);
      int LX_T1110 = (LX_T1109 < (float)0);
      float LX_T1111 = select((float)LX_T1109, (float)0, (int)LX_T1110);
      float LX_T1112 = native_sqrt(LX_T1111);
      X_T1112[gout_idx] = LX_T1112;
    }
  }
}
