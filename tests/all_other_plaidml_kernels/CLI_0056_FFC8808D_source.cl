#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 256 }
// Out stride: { 50176, 3584, 256, 1 }
// Elementwise input X_T331 shape: fp32(1, 14, 14, 256):(50176, 3584, 256, 1):196 KiB
// Elementwise input X_T335 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_132 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T336 = div(X_T331, X_T335)
// Elementwise op: [[pid(Add, Switch)]] X_T337 = add(X_T336, X_I_132)
// Elementwise op: X_T338 = cmp_lt(X_T337, X_T2)
// Elementwise op: [[pid(Relu)]] X_T339 = cond(X_T338, X_T2, X_T337)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 256):(50176, 3584, 256, 1):196 KiB
// Computed true ops: 200704
// Computed work groups: 56
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 8, 1
__kernel void kernel_c29_sdk_77(__global float* restrict  X_T339, __global const float* restrict  X_T331, __global const float* restrict  X_T335, __global const float* restrict  X_I_132)
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
      int gout_idx = (((3584 * (i2_gid + i2_tid)) + (256 * i3)) + (i4_gid + i4_tid));
      float LX_T331 = X_T331[gout_idx];
      float LX_T335 = X_T335[(i4_gid + i4_tid)];
      float LX_I_132 = X_I_132[(i4_gid + i4_tid)];
      float LX_T336 = (LX_T331 / LX_T335);
      float LX_T337 = (LX_T336 + LX_I_132);
      int LX_T338 = (LX_T337 < 0.0f);
      float LX_T339 = select((float)LX_T337, (float)0.0f, (int)LX_T338);
      X_T339[gout_idx] = LX_T339;
    }
  }
}
