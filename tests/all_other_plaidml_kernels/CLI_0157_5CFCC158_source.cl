#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 28 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 160 }
// Out stride: { 125440, 4480, 160, 1 }
// Elementwise input X_T260 shape: fp32(1, 28, 28, 160):(125440, 4480, 160, 1):490 KiB
// Elementwise input X_T264 shape: fp32(160):(1):640 bytes
// Elementwise input X_I_99 shape: fp32(160):(1):640 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T265 = div(X_T260, X_T264)
// Elementwise op: [[pid(Add, Switch)]] X_T266 = add(X_T265, X_I_99)
// Elementwise op: X_T267 = cmp_lt(X_T266, X_T2)
// Elementwise op: [[pid(Relu)]] X_T268 = cond(X_T267, X_T2, X_T266)
// Tile size: { 1, 4, 1, 160 }
// Contraction output var shape: fp32(1, 28, 28, 160):(125440, 4480, 160, 1):490 KiB
// Computed true ops: 501760
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 3072
// Computed mem read: 240
// Computed mem write: 2560
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 28, 1
__kernel void kernel_c68_sdk_77(__global float* restrict  X_T268, __global const float* restrict  X_T260, __global const float* restrict  X_T264, __global const float* restrict  X_I_99)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 3; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 2) || (i4_tid < 32));
    if (i4_cond)
    {
      int i4 = ((64 * i4_lid) + i4_tid);
      int gout_idx = (((4480 * (i2_gid + i2_tid)) + (160 * i3_gid)) + i4);
      float LX_T260 = X_T260[gout_idx];
      float LX_T264 = X_T264[i4];
      float LX_I_99 = X_I_99[i4];
      float LX_T265 = (LX_T260 / LX_T264);
      float LX_T266 = (LX_T265 + LX_I_99);
      int LX_T267 = (LX_T266 < 0.0f);
      float LX_T268 = select((float)LX_T266, (float)0.0f, (int)LX_T267);
      X_T268[gout_idx] = LX_T268;
    }
  }
}
