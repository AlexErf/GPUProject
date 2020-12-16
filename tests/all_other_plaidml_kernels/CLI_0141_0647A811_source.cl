#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 960 }
// Out stride: { 47040, 6720, 960, 1 }
// Elementwise input X_T596 shape: fp32(1, 7, 7, 960):(47040, 6720, 960, 1):183.75 KiB
// Elementwise input X_T600 shape: fp32(960):(1):3.75 KiB
// Elementwise input X_I_237 shape: fp32(960):(1):3.75 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T601 = div(X_T596, X_T600)
// Elementwise op: [[pid(Add, Switch)]] X_T602 = add(X_T601, X_I_237)
// Elementwise op: X_T603 = cmp_lt(X_T602, X_T3)
// Elementwise op: [[pid(Relu)]] X_T604 = cond(X_T603, X_T3, X_T602)
// Elementwise op: X_T605 = cmp_lt(X_T604, X_T2)
// Elementwise op: [[pid(Relu)]] X_T606 = cond(X_T605, X_T604, X_T2)
// Tile size: { 1, 1, 1, 960 }
// Contraction output var shape: fp32(1, 7, 7, 960):(47040, 6720, 960, 1):183.75 KiB
// Computed true ops: 282240
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 360
// Computed mem write: 3840
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c43_sdk_163(__global float* restrict  X_T606, __global const float* restrict  X_T596, __global const float* restrict  X_T600, __global const float* restrict  X_I_237)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_tid < 192));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((6720 * i2_gid) + (960 * i3_gid)) + i4);
      float LX_T596 = X_T596[gout_idx];
      float LX_T600 = X_T600[i4];
      float LX_I_237 = X_I_237[i4];
      float LX_T601 = (LX_T596 / LX_T600);
      float LX_T602 = (LX_T601 + LX_I_237);
      int LX_T603 = (LX_T602 < 0.0f);
      float LX_T604 = select((float)LX_T602, (float)0.0f, (int)LX_T603);
      int LX_T605 = (LX_T604 < 6.0f);
      float LX_T606 = select((float)6.0f, (float)LX_T604, (int)LX_T605);
      X_T606[gout_idx] = LX_T606;
    }
  }
}
