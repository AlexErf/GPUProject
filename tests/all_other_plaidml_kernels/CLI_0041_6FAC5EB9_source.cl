#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 6144 1 1
// lid: 256 1 1
// Names: { i1_i2 }
// Ranges: { 6144 }
// Out stride: { 1 }
// Elementwise input X_I_0 shape: u32(3, 2048):(2048, 1):24 KiB
// Elementwise op: [[pid(PrngState)]] X_T3 = ident(X_I_0)
// Tile size: { 256 }
// Contraction output var shape: u32(3, 2048):(2048, 1):24 KiB
// Computed true ops: 6144
// Computed work groups: 24
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 6144, 1, 1
__kernel void kernel_c25_sdk_0(__global uint* restrict  X_T3, __global const uint* restrict  X_I_0)
{
  int tid = get_local_id(0);
  int i1_i2_gid = (get_group_id(0) * 256);
  int i1_i2_tid = (tid % 256);
  int gout_idx = (i1_i2_gid + i1_i2_tid);
  uint LX_I_0 = X_I_0[gout_idx];
  uint LX_T3 = LX_I_0;
  X_T3[gout_idx] = LX_T3;
}
