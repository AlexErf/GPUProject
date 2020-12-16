#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3840 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1856 }
// Out stride: { 1 }
// Elementwise input X_I_994 shape: fp32(1856):(1):7.25 KiB
// Elementwise op: [[pid(Add)]] X_T2554 = add(X_T71, X_I_994)
// Elementwise op: X_T2555 = cmp_lt(X_T2554, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T2556 = cond(X_T2555, X_T36, X_T2554)
// Elementwise op: [[pid(Sqrt)]] X_T2557 = sqrt(X_T2556)
// Tile size: { 128 }
// Contraction output var shape: fp32(1856):(1):7.25 KiB
// Computed true ops: 7424
// Computed work groups: 15
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 16
// Computed mem write: 512
// Computed operations: 128
// Computed rollups: 0
// Computed threads used: 128
// lwork = 256, 1, 1
// gwork = 3840, 1, 1
__kernel void kernel_c124_sdk_889(__global float* restrict  X_T2557, __global const float* restrict  X_I_994)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 128);
  int i1_tid = (tid % 128);
  int i1_cond = ((i1_gid != 1792) || (i1_tid < 64));
  if (i1_cond)
  {
    if ((tid < 128))
    {
      int gout_idx = (i1_gid + i1_tid);
      float LX_I_994 = X_I_994[gout_idx];
      float LX_T2554 = (1.0009999641624745e-5f + LX_I_994);
      int LX_T2555 = (LX_T2554 < (float)0);
      float LX_T2556 = select((float)LX_T2554, (float)0, (int)LX_T2555);
      float LX_T2557 = native_sqrt(LX_T2556);
      X_T2557[gout_idx] = LX_T2557;
    }
  }
}
