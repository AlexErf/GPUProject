#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 45 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1440 }
// Out stride: { 282240, 20160, 1440, 1 }
// Elementwise input X_T1508 shape: fp32(1, 14, 14, 1440):(282240, 20160, 1440, 1):1102.5 KiB
// Elementwise input X_T1512 shape: fp32(1440):(1):5.625 KiB
// Elementwise input X_I_580 shape: fp32(1440):(1):5.625 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1513 = div(X_T1508, X_T1512)
// Elementwise op: [[pid(Add, Switch)]] X_T1514 = add(X_T1513, X_I_580)
// Elementwise op: X_T1515 = cmp_lt(X_T1514, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1516 = cond(X_T1515, X_T2, X_T1514)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1440):(282240, 20160, 1440, 1):1102.5 KiB
// Computed true ops: 1128960
// Computed work groups: 315
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 45, 1
__kernel void kernel_c124_sdk_515(__global float* restrict  X_T1516, __global const float* restrict  X_T1508, __global const float* restrict  X_T1512, __global const float* restrict  X_I_580)
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
      int gout_idx = (((20160 * (i2_gid + i2_tid)) + (1440 * i3)) + (i4_gid + i4_tid));
      float LX_T1508 = X_T1508[gout_idx];
      float LX_T1512 = X_T1512[(i4_gid + i4_tid)];
      float LX_I_580 = X_I_580[(i4_gid + i4_tid)];
      float LX_T1513 = (LX_T1508 / LX_T1512);
      float LX_T1514 = (LX_T1513 + LX_I_580);
      int LX_T1515 = (LX_T1514 < 0.0f);
      float LX_T1516 = select((float)LX_T1514, (float)0.0f, (int)LX_T1515);
      X_T1516[gout_idx] = LX_T1516;
    }
  }
}
