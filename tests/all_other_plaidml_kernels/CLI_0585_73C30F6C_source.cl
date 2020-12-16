#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 55 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1760 }
// Out stride: { 344960, 24640, 1760, 1 }
// Elementwise input X_T1758 shape: fp32(1, 14, 14, 1760):(344960, 24640, 1760, 1):1347.5 KiB
// Elementwise input X_T1762 shape: fp32(1760):(1):6.875 KiB
// Elementwise input X_I_680 shape: fp32(1760):(1):6.875 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1763 = div(X_T1758, X_T1762)
// Elementwise op: [[pid(Add, Switch)]] X_T1764 = add(X_T1763, X_I_680)
// Elementwise op: X_T1765 = cmp_lt(X_T1764, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1766 = cond(X_T1765, X_T2, X_T1764)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1760):(344960, 24640, 1760, 1):1347.5 KiB
// Computed true ops: 1379840
// Computed work groups: 385
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 55, 1
__kernel void kernel_c124_sdk_605(__global float* restrict  X_T1766, __global const float* restrict  X_T1758, __global const float* restrict  X_T1762, __global const float* restrict  X_I_680)
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
      float LX_T1758 = X_T1758[gout_idx];
      float LX_T1762 = X_T1762[(i4_gid + i4_tid)];
      float LX_I_680 = X_I_680[(i4_gid + i4_tid)];
      float LX_T1763 = (LX_T1758 / LX_T1762);
      float LX_T1764 = (LX_T1763 + LX_I_680);
      int LX_T1765 = (LX_T1764 < 0.0f);
      float LX_T1766 = select((float)LX_T1764, (float)0.0f, (int)LX_T1765);
      X_T1766[gout_idx] = LX_T1766;
    }
  }
}
