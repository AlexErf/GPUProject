#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 14 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 144 }
// Out stride: { 451584, 8064, 144, 1 }
// Elementwise input X_T138 shape: fp32(1, 56, 56, 144):(451584, 8064, 144, 1):1764 KiB
// Elementwise input X_T142 shape: fp32(144):(1):576 bytes
// Elementwise input X_I_105 shape: fp32(144):(1):576 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T143 = div(X_T138, X_T142)
// Elementwise op: [[pid(Add, Switch)]] X_T144 = add(X_T143, X_I_105)
// Elementwise op: X_T145 = cmp_lt(X_T144, X_T3)
// Elementwise op: [[pid(Relu)]] X_T146 = cond(X_T145, X_T3, X_T144)
// Elementwise op: X_T147 = cmp_lt(X_T146, X_T2)
// Elementwise op: [[pid(Relu)]] X_T148 = cond(X_T147, X_T146, X_T2)
// Tile size: { 1, 4, 4, 144 }
// Contraction output var shape: fp32(1, 56, 56, 144):(451584, 8064, 144, 1):1764 KiB
// Computed true ops: 2709504
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 10240
// Computed mem read: 960
// Computed mem write: 10240
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 14, 1
__kernel void kernel_c43_sdk_32(__global float* restrict  X_T148, __global const float* restrict  X_T138, __global const float* restrict  X_T142, __global const float* restrict  X_I_105)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 4);
  int i2_gid = (get_group_id(1) * 4);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 5; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 4) || (i4_tid < 16));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      for (int i2_lid = 0; i2_lid < 2; i2_lid += 1)
      {
        int i2 = ((2 * i2_lid) + i2_tid);
        int gout_idx = (((8064 * (i2_gid + i2)) + (144 * (i3_gid + i3_tid))) + i4);
        float LX_T138 = X_T138[gout_idx];
        float LX_T142 = X_T142[i4];
        float LX_I_105 = X_I_105[i4];
        float LX_T143 = (LX_T138 / LX_T142);
        float LX_T144 = (LX_T143 + LX_I_105);
        int LX_T145 = (LX_T144 < 0.0f);
        float LX_T146 = select((float)LX_T144, (float)0.0f, (int)LX_T145);
        int LX_T147 = (LX_T146 < 6.0f);
        float LX_T148 = select((float)6.0f, (float)LX_T146, (int)LX_T147);
        X_T148[gout_idx] = LX_T148;
      }
    }
  }
}
