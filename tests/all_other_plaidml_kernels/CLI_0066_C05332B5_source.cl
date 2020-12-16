#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 21248 83 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 165, 165, 96 }
// Out stride: { 2613600, 15840, 96, 1 }
// Elementwise input X_T39 shape: fp32(1, 165, 165, 96):(2613600, 15840, 96, 1):10209.4 KiB
// Elementwise input X_T44 shape: fp32(96):(1):384 bytes
// Elementwise input X_I_54 shape: fp32(96):(1):384 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T45 = div(X_T39, X_T44)
// Elementwise op: [[pid(Add, Switch)]] X_T46 = add(X_T45, X_I_54)
// Elementwise op: X_T47 = cmp_lt(X_T46, X_T1)
// Elementwise op: [[pid(Relu)]] X_T48 = cond(X_T47, X_T1, X_T46)
// Tile size: { 1, 2, 2, 96 }
// Contraction output var shape: fp32(1, 165, 165, 96):(2613600, 15840, 96, 1):10209.4 KiB
// Computed true ops: 10454400
// Computed work groups: 6889
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 144
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 21248, 83, 1
__kernel void kernel_c42_sdk_2(__global float* restrict  X_T48, __global const float* restrict  X_T39, __global const float* restrict  X_T44, __global const float* restrict  X_I_54)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  int i3_cond = ((i3_gid != 164) || (i3_tid < 1));
  if (i3_cond)
  {
    int i2_cond = ((i2_gid != 164) || (i2_tid < 1));
    if (i2_cond)
    {
      for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
      {
        int i4_cond = ((i4_lid < 1) || (i4_tid < 32));
        if (i4_cond)
        {
          int i4 = ((64 * i4_lid) + i4_tid);
          int gout_idx = (((15840 * (i2_gid + i2_tid)) + (96 * (i3_gid + i3_tid))) + i4);
          float LX_T39 = X_T39[gout_idx];
          float LX_T44 = X_T44[i4];
          float LX_I_54 = X_I_54[i4];
          float LX_T45 = (LX_T39 / LX_T44);
          float LX_T46 = (LX_T45 + LX_I_54);
          int LX_T47 = (LX_T46 < 0.0f);
          float LX_T48 = select((float)LX_T46, (float)0.0f, (int)LX_T47);
          X_T48[gout_idx] = LX_T48;
        }
      }
    }
  }
}
