#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 83 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 83, 83, 42 }
// Out stride: { 289338, 3486, 42, 1 }
// Elementwise input X_T167 shape: fp32(1, 83, 83, 42):(289338, 3486, 42, 1):1130.23 KiB
// Elementwise input X_T171 shape: fp32(42):(1):168 bytes
// Elementwise input X_I_85 shape: fp32(42):(1):168 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T172 = div(X_T167, X_T171)
// Elementwise op: [[pid(Add, Switch)]] X_T173 = add(X_T172, X_I_85)
// Tile size: { 1, 1, 4, 42 }
// Contraction output var shape: fp32(1, 83, 83, 42):(289338, 3486, 42, 1):1130.23 KiB
// Computed true ops: 578676
// Computed work groups: 1743
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 96
// Computed mem write: 1024
// Computed operations: 168
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 83, 1
__kernel void kernel_c42_sdk_46(__global float* restrict  X_T173, __global const float* restrict  X_T167, __global const float* restrict  X_T171, __global const float* restrict  X_I_85)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 4);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 4);
  int i3_cond = ((i3_gid != 80) || (i3_tid < 3));
  if (i3_cond)
  {
    int i4_cond = (i4_tid < 42);
    if (i4_cond)
    {
      int gout_idx = (((3486 * i2_gid) + (42 * (i3_gid + i3_tid))) + i4_tid);
      float LX_T167 = X_T167[gout_idx];
      float LX_T171 = X_T171[i4_tid];
      float LX_I_85 = X_I_85[i4_tid];
      float LX_T172 = (LX_T167 / LX_T171);
      float LX_T173 = (LX_T172 + LX_I_85);
      X_T173[gout_idx] = LX_T173;
    }
  }
}
