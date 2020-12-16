#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 28 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 112, 112, 64 }
// Out stride: { 802816, 7168, 64, 1 }
// Elementwise input X_T70 shape: fp32(1, 112, 112, 64):(802816, 7168, 64, 1):3136 KiB
// Elementwise input X_T75 shape: fp32(64):(1):256 bytes
// Elementwise input X_I_17 shape: fp32(64):(1):256 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T76 = div(X_T70, X_T75)
// Elementwise op: [[pid(Add, Switch)]] X_T77 = add(X_T76, X_I_17)
// Elementwise op: X_T78 = cmp_lt(X_T77, X_T2)
// Elementwise op: [[pid(Relu)]] X_T79 = cond(X_T78, X_T2, X_T77)
// Tile size: { 1, 4, 16, 64 }
// Contraction output var shape: fp32(1, 112, 112, 64):(802816, 7168, 64, 1):3136 KiB
// Computed true ops: 3211264
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 16384
// Computed mem read: 1536
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 28, 1
__kernel void kernel_c124_sdk_3(__global float* restrict  X_T79, __global const float* restrict  X_T70, __global const float* restrict  X_T75, __global const float* restrict  X_I_17)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 16);
  int i2_gid = (get_group_id(1) * 4);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
  {
    int i4 = ((32 * i4_lid) + i4_tid);
    for (int i3_lid = 0; i3_lid < 4; i3_lid += 1)
    {
      int i3 = ((4 * i3_lid) + i3_tid);
      for (int i2_lid = 0; i2_lid < 2; i2_lid += 1)
      {
        int i2 = ((2 * i2_lid) + i2_tid);
        int gout_idx = (((7168 * (i2_gid + i2)) + (64 * (i3_gid + i3))) + i4);
        float LX_T70 = X_T70[gout_idx];
        float LX_T75 = X_T75[i4];
        float LX_I_17 = X_I_17[i4];
        float LX_T76 = (LX_T70 / LX_T75);
        float LX_T77 = (LX_T76 + LX_I_17);
        int LX_T78 = (LX_T77 < 0.0f);
        float LX_T79 = select((float)LX_T77, (float)0.0f, (int)LX_T78);
        X_T79[gout_idx] = LX_T79;
      }
    }
  }
}
