#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 28 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 416 }
// Out stride: { 326144, 11648, 416, 1 }
// Elementwise input X_T460 shape: fp32(1, 28, 28, 416):(326144, 11648, 416, 1):1274 KiB
// Elementwise input X_T464 shape: fp32(416):(1):1.625 KiB
// Elementwise input X_I_179 shape: fp32(416):(1):1.625 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T465 = div(X_T460, X_T464)
// Elementwise op: [[pid(Add, Switch)]] X_T466 = add(X_T465, X_I_179)
// Elementwise op: X_T467 = cmp_lt(X_T466, X_T2)
// Elementwise op: [[pid(Relu)]] X_T468 = cond(X_T467, X_T2, X_T466)
// Tile size: { 1, 4, 1, 416 }
// Contraction output var shape: fp32(1, 28, 28, 416):(326144, 11648, 416, 1):1274 KiB
// Computed true ops: 1304576
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 624
// Computed mem write: 6656
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 28, 1
__kernel void kernel_c68_sdk_149(__global float* restrict  X_T468, __global const float* restrict  X_T460, __global const float* restrict  X_T464, __global const float* restrict  X_I_179)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 6) || (i4_tid < 32));
    if (i4_cond)
    {
      int i4 = ((64 * i4_lid) + i4_tid);
      int gout_idx = (((11648 * (i2_gid + i2_tid)) + (416 * i3_gid)) + i4);
      float LX_T460 = X_T460[gout_idx];
      float LX_T464 = X_T464[i4];
      float LX_I_179 = X_I_179[i4];
      float LX_T465 = (LX_T460 / LX_T464);
      float LX_T466 = (LX_T465 + LX_I_179);
      int LX_T467 = (LX_T466 < 0.0f);
      float LX_T468 = select((float)LX_T466, (float)0.0f, (int)LX_T467);
      X_T468[gout_idx] = LX_T468;
    }
  }
}
