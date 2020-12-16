#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 48 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 10, 10, 1536 }
// Out stride: { 153600, 15360, 1536, 1 }
// Elementwise input X_T611 shape: fp32(1, 10, 10, 1536):(153600, 15360, 1536, 1):600 KiB
// Elementwise input X_T615 shape: fp32(1536):(1):6 KiB
// Elementwise input X_I_7 shape: fp32(1536):(1):6 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T616 = div(X_T611, X_T615)
// Elementwise op: [[pid(Add, Switch)]] X_T617 = add(X_T616, X_I_7)
// Elementwise op: X_T618 = cmp_lt(X_T617, X_T2)
// Elementwise op: [[pid(Relu)]] X_T619 = cond(X_T618, X_T2, X_T617)
// Tile size: { 1, 2, 10, 32 }
// Contraction output var shape: fp32(1, 10, 10, 1536):(153600, 15360, 1536, 1):600 KiB
// Computed true ops: 614400
// Computed work groups: 240
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 3072
// Computed mem read: 240
// Computed mem write: 2560
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1280, 48, 1
__kernel void kernel_c28_sdk_187(__global float* restrict  X_T619, __global const float* restrict  X_T611, __global const float* restrict  X_T615, __global const float* restrict  X_I_7)
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
      int gout_idx = (((15360 * (i2_gid + i2_tid)) + (1536 * i3)) + (i4_gid + i4_tid));
      float LX_T611 = X_T611[gout_idx];
      float LX_T615 = X_T615[(i4_gid + i4_tid)];
      float LX_I_7 = X_I_7[(i4_gid + i4_tid)];
      float LX_T616 = (LX_T611 / LX_T615);
      float LX_T617 = (LX_T616 + LX_I_7);
      int LX_T618 = (LX_T617 < 0.0f);
      float LX_T619 = select((float)LX_T617, (float)0.0f, (int)LX_T618);
      X_T619[gout_idx] = LX_T619;
    }
  }
}
