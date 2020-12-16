#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 640 }
// Out stride: { 31360, 4480, 640, 1 }
// Elementwise input X_T1395 shape: fp32(1, 7, 7, 640):(31360, 4480, 640, 1):122.5 KiB
// Elementwise input X_T1399 shape: fp32(640):(1):2.5 KiB
// Elementwise input X_I_531 shape: fp32(640):(1):2.5 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1400 = div(X_T1395, X_T1399)
// Elementwise op: [[pid(Add, Switch)]] X_T1401 = add(X_T1400, X_I_531)
// Elementwise op: X_T1402 = cmp_lt(X_T1401, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1403 = cond(X_T1402, X_T2, X_T1401)
// Tile size: { 1, 1, 1, 640 }
// Contraction output var shape: fp32(1, 7, 7, 640):(31360, 4480, 640, 1):122.5 KiB
// Computed true ops: 125440
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 3072
// Computed mem read: 240
// Computed mem write: 2560
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c108_sdk_476(__global float* restrict  X_T1403, __global const float* restrict  X_T1395, __global const float* restrict  X_T1399, __global const float* restrict  X_I_531)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 3; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 2) || (i4_tid < 128));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((4480 * i2_gid) + (640 * i3_gid)) + i4);
      float LX_T1395 = X_T1395[gout_idx];
      float LX_T1399 = X_T1399[i4];
      float LX_I_531 = X_I_531[i4];
      float LX_T1400 = (LX_T1395 / LX_T1399);
      float LX_T1401 = (LX_T1400 + LX_I_531);
      int LX_T1402 = (LX_T1401 < 0.0f);
      float LX_T1403 = select((float)LX_T1401, (float)0.0f, (int)LX_T1402);
      X_T1403[gout_idx] = LX_T1403;
    }
  }
}
