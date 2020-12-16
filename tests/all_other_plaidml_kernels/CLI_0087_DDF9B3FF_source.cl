#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 4864 19 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 19, 19, 1024 }
// Out stride: { 369664, 19456, 1024, 1 }
// Elementwise input X_T585 shape: fp32(1, 19, 19, 1024):(369664, 19456, 1024, 1):1444 KiB
// Elementwise input X_T589 shape: fp32(1024):(1):4 KiB
// Elementwise input X_I_12 shape: fp32(1024):(1):4 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T590 = div(X_T585, X_T589)
// Elementwise op: [[pid(Add, Switch)]] X_T591 = add(X_T590, X_I_12)
// Tile size: { 1, 1, 1, 1024 }
// Contraction output var shape: fp32(1, 19, 19, 1024):(369664, 19456, 1024, 1):1444 KiB
// Computed true ops: 739328
// Computed work groups: 361
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 384
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 4864, 19, 1
__kernel void kernel_c28_sdk_179(__global float* restrict  X_T591, __global const float* restrict  X_T585, __global const float* restrict  X_T589, __global const float* restrict  X_I_12)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4 = ((256 * i4_lid) + i4_tid);
    int gout_idx = (((19456 * i2_gid) + (1024 * i3_gid)) + i4);
    float LX_T585 = X_T585[gout_idx];
    float LX_T589 = X_T589[i4];
    float LX_I_12 = X_I_12[i4];
    float LX_T590 = (LX_T585 / LX_T589);
    float LX_T591 = (LX_T590 + LX_I_12);
    X_T591[gout_idx] = LX_T591;
  }
}
