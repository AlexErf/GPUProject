#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 56 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 256 }
// Out stride: { 802816, 14336, 256, 1 }
// Elementwise input X_T76 shape: fp32(1, 56, 56, 256):(802816, 14336, 256, 1):3136 KiB
// Elementwise input X_T80 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_227 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T81 = div(X_T76, X_T80)
// Elementwise op: [[pid(Add, Switch)]] X_T82 = add(X_T81, X_I_227)
// Tile size: { 1, 4, 1, 256 }
// Contraction output var shape: fp32(1, 56, 56, 256):(802816, 14336, 256, 1):3136 KiB
// Computed true ops: 1605632
// Computed work groups: 784
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 384
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 56, 1
__kernel void kernel_c29_sdk_14(__global float* restrict  X_T82, __global const float* restrict  X_T76, __global const float* restrict  X_T80, __global const float* restrict  X_I_227)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4 = ((64 * i4_lid) + i4_tid);
    int gout_idx = (((14336 * (i2_gid + i2_tid)) + (256 * i3_gid)) + i4);
    float LX_T76 = X_T76[gout_idx];
    float LX_T80 = X_T80[i4];
    float LX_I_227 = X_I_227[i4];
    float LX_T81 = (LX_T76 / LX_T80);
    float LX_T82 = (LX_T81 + LX_I_227);
    X_T82[gout_idx] = LX_T82;
  }
}
