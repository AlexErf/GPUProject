#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 38 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1216 }
// Out stride: { 238336, 17024, 1216, 1 }
// Elementwise input X_T1298 shape: fp32(1, 14, 14, 1216):(238336, 17024, 1216, 1):931 KiB
// Elementwise input X_T1321 shape: fp32(1, 14, 14, 1216):(238336, 17024, 1216, 1):931 KiB
// Elementwise input X_I_512 shape: fp32(1216):(1):4.75 KiB
// Elementwise input X_I_511 shape: fp32(1216):(1):4.75 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1322 = add(X_T1298, X_T1321)
// Elementwise op: [[pid(Sub)]] X_T1324 = sub(X_T1322, X_I_512)
// Elementwise op: [[pid(Mul)]] X_T1325 = mul(X_T1324, X_I_511)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1216):(238336, 17024, 1216, 1):931 KiB
// Computed true ops: 715008
// Computed work groups: 266
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 38, 1
__kernel void kernel_c108_sdk_449(__global float* restrict  X_T1322, __global float* restrict  X_T1325, __global const float* restrict  X_T1298, __global const float* restrict  X_T1321, __global const float* restrict  X_I_512, __global const float* restrict  X_I_511)
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
      int gout_idx = (((17024 * (i2_gid + i2_tid)) + (1216 * i3)) + (i4_gid + i4_tid));
      float LX_T1298 = X_T1298[gout_idx];
      float LX_T1321 = X_T1321[gout_idx];
      float LX_I_512 = X_I_512[(i4_gid + i4_tid)];
      float LX_I_511 = X_I_511[(i4_gid + i4_tid)];
      float LX_T1322 = (LX_T1298 + LX_T1321);
      float LX_T1324 = (LX_T1322 - LX_I_512);
      float LX_T1325 = (LX_T1324 * LX_I_511);
      X_T1322[gout_idx] = LX_T1322;
      X_T1325[gout_idx] = LX_T1325;
    }
  }
}
