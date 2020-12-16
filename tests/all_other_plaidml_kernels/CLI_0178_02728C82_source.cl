#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 9 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 288 }
// Out stride: { 225792, 8064, 288, 1 }
// Elementwise input X_T360 shape: fp32(1, 28, 28, 288):(225792, 8064, 288, 1):882 KiB
// Elementwise input X_T364 shape: fp32(288):(1):1.125 KiB
// Elementwise input X_I_139 shape: fp32(288):(1):1.125 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T365 = div(X_T360, X_T364)
// Elementwise op: [[pid(Add, Switch)]] X_T366 = add(X_T365, X_I_139)
// Elementwise op: X_T367 = cmp_lt(X_T366, X_T2)
// Elementwise op: [[pid(Relu)]] X_T368 = cond(X_T367, X_T2, X_T366)
// Tile size: { 1, 4, 28, 32 }
// Contraction output var shape: fp32(1, 28, 28, 288):(225792, 8064, 288, 1):882 KiB
// Computed true ops: 903168
// Computed work groups: 63
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1344
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 9, 1
__kernel void kernel_c68_sdk_113(__global float* restrict  X_T368, __global const float* restrict  X_T360, __global const float* restrict  X_T364, __global const float* restrict  X_I_139)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 7; i3_lid += 1)
  {
    int i3 = ((4 * i3_lid) + i3_tid);
    for (int i2_lid = 0; i2_lid < 2; i2_lid += 1)
    {
      int i2 = ((2 * i2_lid) + i2_tid);
      int gout_idx = (((8064 * (i2_gid + i2)) + (288 * i3)) + (i4_gid + i4_tid));
      float LX_T360 = X_T360[gout_idx];
      float LX_T364 = X_T364[(i4_gid + i4_tid)];
      float LX_I_139 = X_I_139[(i4_gid + i4_tid)];
      float LX_T365 = (LX_T360 / LX_T364);
      float LX_T366 = (LX_T365 + LX_I_139);
      int LX_T367 = (LX_T366 < 0.0f);
      float LX_T368 = select((float)LX_T366, (float)0.0f, (int)LX_T367);
      X_T368[gout_idx] = LX_T368;
    }
  }
}
