#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 160 }
// Out stride: { 7840, 1120, 160, 1 }
// Elementwise input X_T586 shape: fp32(1, 7, 7, 160):(7840, 1120, 160, 1):30.625 KiB
// Elementwise input X_T590 shape: fp32(160):(1):640 bytes
// Elementwise input X_I_18 shape: fp32(160):(1):640 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T591 = div(X_T586, X_T590)
// Elementwise op: [[pid(Add, Switch)]] X_T592 = add(X_T591, X_I_18)
// Tile size: { 1, 7, 1, 32 }
// Contraction output var shape: fp32(1, 7, 7, 160):(7840, 1120, 160, 1):30.625 KiB
// Computed true ops: 15680
// Computed work groups: 35
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 84
// Computed mem write: 896
// Computed operations: 224
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1280, 7, 1
__kernel void kernel_c43_sdk_159(__global float* restrict  X_T592, __global const float* restrict  X_T586, __global const float* restrict  X_T590, __global const float* restrict  X_I_18)
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
    float LX_T586 = X_T586[gout_idx];
    float LX_T590 = X_T590[(i4_gid + i4_tid)];
    float LX_I_18 = X_I_18[(i4_gid + i4_tid)];
    float LX_T591 = (LX_T586 / LX_T590);
    float LX_T592 = (LX_T591 + LX_I_18);
    X_T592[gout_idx] = LX_T592;
  }
}
