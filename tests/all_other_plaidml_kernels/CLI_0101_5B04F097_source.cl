#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 32 }
// Out stride: { 25088, 896, 32, 1 }
// Elementwise input X_T204 shape: fp32(1, 28, 28, 32):(25088, 896, 32, 1):98 KiB
// Elementwise input X_T208 shape: fp32(32):(1):128 bytes
// Elementwise input X_I_54 shape: fp32(32):(1):128 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T209 = div(X_T204, X_T208)
// Elementwise op: [[pid(Add, Switch)]] X_T210 = add(X_T209, X_I_54)
// Tile size: { 1, 28, 1, 32 }
// Contraction output var shape: fp32(1, 28, 28, 32):(25088, 896, 32, 1):98 KiB
// Computed true ops: 50176
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c43_sdk_50(__global float* restrict  X_T210, __global const float* restrict  X_T204, __global const float* restrict  X_T208, __global const float* restrict  X_I_54)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i2_tid = ((tid / 32) % 8);
  for (int i2_lid = 0; i2_lid < 4; i2_lid += 1)
  {
    int i2_cond = ((i2_lid < 3) || (i2_tid < 4));
    if (i2_cond)
    {
      int i2 = ((8 * i2_lid) + i2_tid);
      int gout_idx = (((896 * i2) + (32 * i3_gid)) + i4_tid);
      float LX_T204 = X_T204[gout_idx];
      float LX_T208 = X_T208[i4_tid];
      float LX_I_54 = X_I_54[i4_tid];
      float LX_T209 = (LX_T204 / LX_T208);
      float LX_T210 = (LX_T209 + LX_I_54);
      X_T210[gout_idx] = LX_T210;
    }
  }
}
