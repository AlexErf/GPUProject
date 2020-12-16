#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 160 }
// Out stride: { 7840, 1120, 160, 1 }
// Elementwise input X_T623 shape: fp32(1, 7, 7, 160):(7840, 1120, 160, 1):30.625 KiB
// Elementwise input X_T627 shape: fp32(160):(1):640 bytes
// Elementwise input X_I_229 shape: fp32(160):(1):640 bytes
// Elementwise input X_T592 shape: fp32(1, 7, 7, 160):(7840, 1120, 160, 1):30.625 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T628 = div(X_T623, X_T627)
// Elementwise op: [[pid(Add, Switch)]] X_T629 = add(X_T628, X_I_229)
// Elementwise op: [[pid(Add)]] X_T630 = add(X_T592, X_T629)
// Tile size: { 1, 7, 1, 32 }
// Contraction output var shape: fp32(1, 7, 7, 160):(7840, 1120, 160, 1):30.625 KiB
// Computed true ops: 23520
// Computed work groups: 35
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 112
// Computed mem write: 896
// Computed operations: 224
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1280, 7, 1
__kernel void kernel_c43_sdk_170(__global float* restrict  X_T630, __global const float* restrict  X_T623, __global const float* restrict  X_T627, __global const float* restrict  X_I_229, __global const float* restrict  X_T592)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 32);
  int i3_gid = get_group_id(1);
  int i4_tid = (tid % 32);
  int i2_tid = ((tid / 32) % 8);
  int i2_cond = (i2_tid < 7);
  if (i2_cond)
  {
    int gout_idx = (((1120 * i2_tid) + (160 * i3_gid)) + (i4_gid + i4_tid));
    float LX_T623 = X_T623[gout_idx];
    float LX_T627 = X_T627[(i4_gid + i4_tid)];
    float LX_I_229 = X_I_229[(i4_gid + i4_tid)];
    float LX_T592 = X_T592[gout_idx];
    float LX_T628 = (LX_T623 / LX_T627);
    float LX_T629 = (LX_T628 + LX_I_229);
    float LX_T630 = (LX_T592 + LX_T629);
    X_T630[gout_idx] = LX_T630;
  }
}
