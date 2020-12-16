#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 64 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 10, 10, 2048 }
// Out stride: { 204800, 20480, 2048, 1 }
// Elementwise input X_T624 shape: fp32(1, 10, 10, 2048):(204800, 20480, 2048, 1):800 KiB
// Elementwise input X_T628 shape: fp32(2048):(1):8 KiB
// Elementwise input X_I_2 shape: fp32(2048):(1):8 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T629 = div(X_T624, X_T628)
// Elementwise op: [[pid(Add, Switch)]] X_T630 = add(X_T629, X_I_2)
// Elementwise op: X_T631 = cmp_lt(X_T630, X_T2)
// Elementwise op: [[pid(Relu)]] X_T632 = cond(X_T631, X_T2, X_T630)
// Tile size: { 1, 2, 10, 32 }
// Contraction output var shape: fp32(1, 10, 10, 2048):(204800, 20480, 2048, 1):800 KiB
// Computed true ops: 819200
// Computed work groups: 320
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 3072
// Computed mem read: 240
// Computed mem write: 2560
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1280, 64, 1
__kernel void kernel_c28_sdk_191(__global float* restrict  X_T632, __global const float* restrict  X_T624, __global const float* restrict  X_T628, __global const float* restrict  X_I_2)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i2_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 3; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 2) || (i3_tid < 2));
    if (i3_cond)
    {
      int i3 = ((4 * i3_lid) + i3_tid);
      int gout_idx = (((20480 * (i2_gid + i2_tid)) + (2048 * i3)) + (i4_gid + i4_tid));
      float LX_T624 = X_T624[gout_idx];
      float LX_T628 = X_T628[(i4_gid + i4_tid)];
      float LX_I_2 = X_I_2[(i4_gid + i4_tid)];
      float LX_T629 = (LX_T624 / LX_T628);
      float LX_T630 = (LX_T629 + LX_I_2);
      int LX_T631 = (LX_T630 < 0.0f);
      float LX_T632 = select((float)LX_T630, (float)0.0f, (int)LX_T631);
      X_T632[gout_idx] = LX_T632;
    }
  }
}
