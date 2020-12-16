#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 9472 37 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 74, 74, 128 }
// Out stride: { 700928, 9472, 128, 1 }
// Elementwise input X_T156 shape: fp32(1, 74, 74, 128):(700928, 9472, 128, 1):2738 KiB
// Elementwise input X_T160 shape: fp32(128):(1):512 bytes
// Elementwise input X_I_185 shape: fp32(128):(1):512 bytes
// Elementwise input X_T149 shape: fp32(1, 74, 74, 128):(700928, 9472, 128, 1):2738 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T161 = div(X_T156, X_T160)
// Elementwise op: [[pid(Add, Switch)]] X_T162 = add(X_T161, X_I_185)
// Elementwise op: [[pid(Add)]] X_T163 = add(X_T149, X_T162)
// Elementwise op: X_T164 = cmp_lt(X_T163, X_T2)
// Elementwise op: [[pid(Relu)]] X_T165 = cond(X_T164, X_T2, X_T163)
// Tile size: { 1, 2, 2, 128 }
// Contraction output var shape: fp32(1, 74, 74, 128):(700928, 9472, 128, 1):2738 KiB
// Computed true ops: 3504640
// Computed work groups: 1369
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 256
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 9472, 37, 1
__kernel void kernel_c28_sdk_51(__global float* restrict  X_T163, __global float* restrict  X_T165, __global const float* restrict  X_T156, __global const float* restrict  X_T160, __global const float* restrict  X_I_185, __global const float* restrict  X_T149)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
  {
    int i4 = ((64 * i4_lid) + i4_tid);
    int gout_idx = (((9472 * (i2_gid + i2_tid)) + (128 * (i3_gid + i3_tid))) + i4);
    float LX_T156 = X_T156[gout_idx];
    float LX_T160 = X_T160[i4];
    float LX_I_185 = X_I_185[i4];
    float LX_T149 = X_T149[gout_idx];
    float LX_T161 = (LX_T156 / LX_T160);
    float LX_T162 = (LX_T161 + LX_I_185);
    float LX_T163 = (LX_T149 + LX_T162);
    int LX_T164 = (LX_T163 < 0.0f);
    float LX_T165 = select((float)LX_T163, (float)0.0f, (int)LX_T164);
    X_T163[gout_idx] = LX_T163;
    X_T165[gout_idx] = LX_T165;
  }
}
