#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 16 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 2048 }
// Out stride: { 100352, 14336, 2048, 1 }
// Elementwise input X_T597 shape: fp32(1, 7, 7, 2048):(100352, 14336, 2048, 1):392 KiB
// Elementwise input X_T601 shape: fp32(2048):(1):8 KiB
// Elementwise input X_I_309 shape: fp32(2048):(1):8 KiB
// Elementwise input X_T593 shape: fp32(1, 7, 7, 2048):(100352, 14336, 2048, 1):392 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T602 = div(X_T597, X_T601)
// Elementwise op: [[pid(Add, Switch)]] X_T603 = add(X_T602, X_I_309)
// Elementwise op: [[pid(Add)]] X_T604 = add(X_T593, X_T603)
// Elementwise op: X_T605 = cmp_lt(X_T604, X_T2)
// Elementwise op: [[pid(Relu)]] X_T606 = cond(X_T605, X_T2, X_T604)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 2048):(100352, 14336, 2048, 1):392 KiB
// Computed true ops: 501760
// Computed work groups: 112
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 16, 1
__kernel void kernel_c29_sdk_143(__global float* restrict  X_T606, __global const float* restrict  X_T597, __global const float* restrict  X_T601, __global const float* restrict  X_I_309, __global const float* restrict  X_T593)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  int i3_cond = (i3_tid < 7);
  if (i3_cond)
  {
    for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int gout_idx = (((14336 * i2_gid) + (2048 * i3_tid)) + (i4_gid + i4));
      float LX_T597 = X_T597[gout_idx];
      float LX_T601 = X_T601[(i4_gid + i4)];
      float LX_I_309 = X_I_309[(i4_gid + i4)];
      float LX_T593 = X_T593[gout_idx];
      float LX_T602 = (LX_T597 / LX_T601);
      float LX_T603 = (LX_T602 + LX_I_309);
      float LX_T604 = (LX_T593 + LX_T603);
      int LX_T605 = (LX_T604 < 0.0f);
      float LX_T606 = select((float)LX_T604, (float)0.0f, (int)LX_T605);
      X_T606[gout_idx] = LX_T606;
    }
  }
}
