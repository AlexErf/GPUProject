#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 32 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1024 }
// Out stride: { 200704, 14336, 1024, 1 }
// Elementwise input X_T1148 shape: fp32(1, 14, 14, 1024):(200704, 14336, 1024, 1):784 KiB
// Elementwise input X_T1171 shape: fp32(1, 14, 14, 1024):(200704, 14336, 1024, 1):784 KiB
// Elementwise input X_I_452 shape: fp32(1024):(1):4 KiB
// Elementwise input X_I_451 shape: fp32(1024):(1):4 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1172 = add(X_T1148, X_T1171)
// Elementwise op: [[pid(Sub)]] X_T1174 = sub(X_T1172, X_I_452)
// Elementwise op: [[pid(Mul)]] X_T1175 = mul(X_T1174, X_I_451)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1024):(200704, 14336, 1024, 1):784 KiB
// Computed true ops: 602112
// Computed work groups: 224
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 32, 1
__kernel void kernel_c108_sdk_395(__global float* restrict  X_T1172, __global float* restrict  X_T1175, __global const float* restrict  X_T1148, __global const float* restrict  X_T1171, __global const float* restrict  X_I_452, __global const float* restrict  X_I_451)
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
      float LX_T1148 = X_T1148[gout_idx];
      float LX_T1171 = X_T1171[gout_idx];
      float LX_I_452 = X_I_452[(i4_gid + i4_tid)];
      float LX_I_451 = X_I_451[(i4_gid + i4_tid)];
      float LX_T1172 = (LX_T1148 + LX_T1171);
      float LX_T1174 = (LX_T1172 - LX_I_452);
      float LX_T1175 = (LX_T1174 * LX_I_451);
      X_T1172[gout_idx] = LX_T1172;
      X_T1175[gout_idx] = LX_T1175;
    }
  }
}
