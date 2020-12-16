#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 55 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1760 }
// Out stride: { 344960, 24640, 1760, 1 }
// Elementwise input X_T1731 shape: fp32(1, 14, 14, 1760):(344960, 24640, 1760, 1):1347.5 KiB
// Elementwise input X_T1754 shape: fp32(1, 14, 14, 1760):(344960, 24640, 1760, 1):1347.5 KiB
// Elementwise input X_I_682 shape: fp32(1760):(1):6.875 KiB
// Elementwise input X_I_681 shape: fp32(1760):(1):6.875 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1755 = add(X_T1731, X_T1754)
// Elementwise op: [[pid(Sub)]] X_T1757 = sub(X_T1755, X_I_682)
// Elementwise op: [[pid(Mul)]] X_T1758 = mul(X_T1757, X_I_681)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1760):(344960, 24640, 1760, 1):1347.5 KiB
// Computed true ops: 1034880
// Computed work groups: 385
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 55, 1
__kernel void kernel_c124_sdk_602(__global float* restrict  X_T1755, __global float* restrict  X_T1758, __global const float* restrict  X_T1731, __global const float* restrict  X_T1754, __global const float* restrict  X_I_682, __global const float* restrict  X_I_681)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i2_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 4; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 3) || (i3_tid < 2));
    if (i3_cond)
    {
      int i3 = ((4 * i3_lid) + i3_tid);
      int gout_idx = (((24640 * (i2_gid + i2_tid)) + (1760 * i3)) + (i4_gid + i4_tid));
      float LX_T1731 = X_T1731[gout_idx];
      float LX_T1754 = X_T1754[gout_idx];
      float LX_I_682 = X_I_682[(i4_gid + i4_tid)];
      float LX_I_681 = X_I_681[(i4_gid + i4_tid)];
      float LX_T1755 = (LX_T1731 + LX_T1754);
      float LX_T1757 = (LX_T1755 - LX_I_682);
      float LX_T1758 = (LX_T1757 * LX_I_681);
      X_T1755[gout_idx] = LX_T1755;
      X_T1758[gout_idx] = LX_T1758;
    }
  }
}
