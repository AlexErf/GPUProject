#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 32 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1024 }
// Out stride: { 200704, 14336, 1024, 1 }
// Elementwise input X_T1154 shape: fp32(1, 14, 14, 1024):(200704, 14336, 1024, 1):784 KiB
// Elementwise input X_T1158 shape: fp32(1024):(1):4 KiB
// Elementwise input X_I_6 shape: fp32(1024):(1):4 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1159 = div(X_T1154, X_T1158)
// Elementwise op: [[pid(Add, Switch)]] X_T1160 = add(X_T1159, X_I_6)
// Elementwise op: X_T1161 = cmp_lt(X_T1160, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1162 = cond(X_T1161, X_T2, X_T1160)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1024):(200704, 14336, 1024, 1):784 KiB
// Computed true ops: 802816
// Computed work groups: 224
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 32, 1
__kernel void kernel_c68_sdk_397(__global float* restrict  X_T1162, __global const float* restrict  X_T1154, __global const float* restrict  X_T1158, __global const float* restrict  X_I_6)
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
      int gout_idx = (((14336 * (i2_gid + i2_tid)) + (1024 * i3)) + (i4_gid + i4_tid));
      float LX_T1154 = X_T1154[gout_idx];
      float LX_T1158 = X_T1158[(i4_gid + i4_tid)];
      float LX_I_6 = X_I_6[(i4_gid + i4_tid)];
      float LX_T1159 = (LX_T1154 / LX_T1158);
      float LX_T1160 = (LX_T1159 + LX_I_6);
      int LX_T1161 = (LX_T1160 < 0.0f);
      float LX_T1162 = select((float)LX_T1160, (float)0.0f, (int)LX_T1161);
      X_T1162[gout_idx] = LX_T1162;
    }
  }
}
