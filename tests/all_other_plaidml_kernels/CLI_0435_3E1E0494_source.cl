#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 30 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 960 }
// Out stride: { 188160, 13440, 960, 1 }
// Elementwise input X_T1133 shape: fp32(1, 14, 14, 960):(188160, 13440, 960, 1):735 KiB
// Elementwise input X_T1137 shape: fp32(960):(1):3.75 KiB
// Elementwise input X_I_430 shape: fp32(960):(1):3.75 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1138 = div(X_T1133, X_T1137)
// Elementwise op: [[pid(Add, Switch)]] X_T1139 = add(X_T1138, X_I_430)
// Elementwise op: X_T1140 = cmp_lt(X_T1139, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1141 = cond(X_T1140, X_T2, X_T1139)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 960):(188160, 13440, 960, 1):735 KiB
// Computed true ops: 752640
// Computed work groups: 210
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 30, 1
__kernel void kernel_c124_sdk_380(__global float* restrict  X_T1141, __global const float* restrict  X_T1133, __global const float* restrict  X_T1137, __global const float* restrict  X_I_430)
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
      int gout_idx = (((13440 * (i2_gid + i2_tid)) + (960 * i3)) + (i4_gid + i4_tid));
      float LX_T1133 = X_T1133[gout_idx];
      float LX_T1137 = X_T1137[(i4_gid + i4_tid)];
      float LX_I_430 = X_I_430[(i4_gid + i4_tid)];
      float LX_T1138 = (LX_T1133 / LX_T1137);
      float LX_T1139 = (LX_T1138 + LX_I_430);
      int LX_T1140 = (LX_T1139 < 0.0f);
      float LX_T1141 = select((float)LX_T1139, (float)0.0f, (int)LX_T1140);
      X_T1141[gout_idx] = LX_T1141;
    }
  }
}
