#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 416 }
// Out stride: { 81536, 5824, 416, 1 }
// Elementwise input X_T680 shape: fp32(1, 14, 14, 416):(81536, 5824, 416, 1):318.5 KiB
// Elementwise input X_T684 shape: fp32(416):(1):1.625 KiB
// Elementwise input X_I_260 shape: fp32(416):(1):1.625 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T685 = div(X_T680, X_T684)
// Elementwise op: [[pid(Add, Switch)]] X_T686 = add(X_T685, X_I_260)
// Elementwise op: X_T687 = cmp_lt(X_T686, X_T2)
// Elementwise op: [[pid(Relu)]] X_T688 = cond(X_T687, X_T2, X_T686)
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
__kernel void kernel_c68_sdk_227(__global float* restrict  X_T688, __global const float* restrict  X_T680, __global const float* restrict  X_T684, __global const float* restrict  X_I_260)
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
      float LX_T680 = X_T680[gout_idx];
      float LX_T684 = X_T684[i4];
      float LX_I_260 = X_I_260[i4];
      float LX_T685 = (LX_T680 / LX_T684);
      float LX_T686 = (LX_T685 + LX_I_260);
      int LX_T687 = (LX_T686 < 0.0f);
      float LX_T688 = select((float)LX_T686, (float)0.0f, (int)LX_T687);
      X_T688[gout_idx] = LX_T688;
    }
  }
}
