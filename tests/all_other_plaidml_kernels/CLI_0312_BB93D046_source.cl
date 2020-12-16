#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 416 }
// Out stride: { 81536, 5824, 416, 1 }
// Elementwise input X_T700 shape: fp32(1, 14, 14, 416):(81536, 5824, 416, 1):318.5 KiB
// Elementwise input X_T704 shape: fp32(416):(1):1.625 KiB
// Elementwise input X_I_260 shape: fp32(416):(1):1.625 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T705 = div(X_T700, X_T704)
// Elementwise op: [[pid(Add, Switch)]] X_T706 = add(X_T705, X_I_260)
// Elementwise op: X_T707 = cmp_lt(X_T706, X_T2)
// Elementwise op: [[pid(Relu)]] X_T708 = cond(X_T707, X_T2, X_T706)
// Tile size: { 1, 2, 2, 416 }
// Contraction output var shape: fp32(1, 14, 14, 416):(81536, 5824, 416, 1):318.5 KiB
// Computed true ops: 326144
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 624
// Computed mem write: 6656
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c108_sdk_227(__global float* restrict  X_T708, __global const float* restrict  X_T700, __global const float* restrict  X_T704, __global const float* restrict  X_I_260)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 6) || (i4_tid < 32));
    if (i4_cond)
    {
      int i4 = ((64 * i4_lid) + i4_tid);
      int gout_idx = (((5824 * (i2_gid + i2_tid)) + (416 * (i3_gid + i3_tid))) + i4);
      float LX_T700 = X_T700[gout_idx];
      float LX_T704 = X_T704[i4];
      float LX_I_260 = X_I_260[i4];
      float LX_T705 = (LX_T700 / LX_T704);
      float LX_T706 = (LX_T705 + LX_I_260);
      int LX_T707 = (LX_T706 < 0.0f);
      float LX_T708 = select((float)LX_T706, (float)0.0f, (int)LX_T707);
      X_T708[gout_idx] = LX_T708;
    }
  }
}
