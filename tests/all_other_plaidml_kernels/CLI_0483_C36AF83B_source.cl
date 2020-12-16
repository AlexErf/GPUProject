#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 38 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1216 }
// Out stride: { 238336, 17024, 1216, 1 }
// Elementwise input X_T1333 shape: fp32(1, 14, 14, 1216):(238336, 17024, 1216, 1):931 KiB
// Elementwise input X_T1337 shape: fp32(1216):(1):4.75 KiB
// Elementwise input X_I_510 shape: fp32(1216):(1):4.75 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1338 = div(X_T1333, X_T1337)
// Elementwise op: [[pid(Add, Switch)]] X_T1339 = add(X_T1338, X_I_510)
// Elementwise op: X_T1340 = cmp_lt(X_T1339, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1341 = cond(X_T1340, X_T2, X_T1339)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1216):(238336, 17024, 1216, 1):931 KiB
// Computed true ops: 953344
// Computed work groups: 266
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 38, 1
__kernel void kernel_c124_sdk_452(__global float* restrict  X_T1341, __global const float* restrict  X_T1333, __global const float* restrict  X_T1337, __global const float* restrict  X_I_510)
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
      int gout_idx = (((17024 * (i2_gid + i2_tid)) + (1216 * i3)) + (i4_gid + i4_tid));
      float LX_T1333 = X_T1333[gout_idx];
      float LX_T1337 = X_T1337[(i4_gid + i4_tid)];
      float LX_I_510 = X_I_510[(i4_gid + i4_tid)];
      float LX_T1338 = (LX_T1333 / LX_T1337);
      float LX_T1339 = (LX_T1338 + LX_I_510);
      int LX_T1340 = (LX_T1339 < 0.0f);
      float LX_T1341 = select((float)LX_T1339, (float)0.0f, (int)LX_T1340);
      X_T1341[gout_idx] = LX_T1341;
    }
  }
}
