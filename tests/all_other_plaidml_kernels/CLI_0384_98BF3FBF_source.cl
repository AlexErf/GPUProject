#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 22 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 704 }
// Out stride: { 137984, 9856, 704, 1 }
// Elementwise input X_T906 shape: fp32(1, 14, 14, 704):(137984, 9856, 704, 1):539 KiB
// Elementwise input X_T929 shape: fp32(1, 14, 14, 704):(137984, 9856, 704, 1):539 KiB
// Elementwise input X_I_352 shape: fp32(704):(1):2.75 KiB
// Elementwise input X_I_351 shape: fp32(704):(1):2.75 KiB
// Elementwise op: [[pid(Concatenate)]] X_T930 = add(X_T906, X_T929)
// Elementwise op: [[pid(Sub)]] X_T932 = sub(X_T930, X_I_352)
// Elementwise op: [[pid(Mul)]] X_T933 = mul(X_T932, X_I_351)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 704):(137984, 9856, 704, 1):539 KiB
// Computed true ops: 413952
// Computed work groups: 154
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 22, 1
__kernel void kernel_c124_sdk_305(__global float* restrict  X_T930, __global float* restrict  X_T933, __global const float* restrict  X_T906, __global const float* restrict  X_T929, __global const float* restrict  X_I_352, __global const float* restrict  X_I_351)
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
      int gout_idx = (((9856 * (i2_gid + i2_tid)) + (704 * i3)) + (i4_gid + i4_tid));
      float LX_T906 = X_T906[gout_idx];
      float LX_T929 = X_T929[gout_idx];
      float LX_I_352 = X_I_352[(i4_gid + i4_tid)];
      float LX_I_351 = X_I_351[(i4_gid + i4_tid)];
      float LX_T930 = (LX_T906 + LX_T929);
      float LX_T932 = (LX_T930 - LX_I_352);
      float LX_T933 = (LX_T932 * LX_I_351);
      X_T930[gout_idx] = LX_T930;
      X_T933[gout_idx] = LX_T933;
    }
  }
}
