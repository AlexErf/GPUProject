#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 640 }
// Out stride: { 31360, 4480, 640, 1 }
// Elementwise input X_T1248 shape: fp32(1, 7, 7, 640):(31360, 4480, 640, 1):122.5 KiB
// Elementwise input X_T1271 shape: fp32(1, 7, 7, 640):(31360, 4480, 640, 1):122.5 KiB
// Elementwise input X_I_493 shape: fp32(640):(1):2.5 KiB
// Elementwise input X_I_492 shape: fp32(640):(1):2.5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1272 = add(X_T1248, X_T1271)
// Elementwise op: [[pid(Sub)]] X_T1274 = sub(X_T1272, X_I_493)
// Elementwise op: [[pid(Mul)]] X_T1275 = mul(X_T1274, X_I_492)
// Tile size: { 1, 1, 1, 640 }
// Contraction output var shape: fp32(1, 7, 7, 640):(31360, 4480, 640, 1):122.5 KiB
// Computed true ops: 94080
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 3072
// Computed mem read: 320
// Computed mem write: 5120
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c68_sdk_437(__global float* restrict  X_T1272, __global float* restrict  X_T1275, __global const float* restrict  X_T1248, __global const float* restrict  X_T1271, __global const float* restrict  X_I_493, __global const float* restrict  X_I_492)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 3; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 2) || (i4_tid < 128));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((4480 * i2_gid) + (640 * i3_gid)) + i4);
      float LX_T1248 = X_T1248[gout_idx];
      float LX_T1271 = X_T1271[gout_idx];
      float LX_I_493 = X_I_493[i4];
      float LX_I_492 = X_I_492[i4];
      float LX_T1272 = (LX_T1248 + LX_T1271);
      float LX_T1274 = (LX_T1272 - LX_I_493);
      float LX_T1275 = (LX_T1274 * LX_I_492);
      X_T1272[gout_idx] = LX_T1272;
      X_T1275[gout_idx] = LX_T1275;
    }
  }
}
