#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1760 }
// Out stride: { 86240, 12320, 1760, 1 }
// Elementwise input X_T2451 shape: fp32(1, 7, 7, 1760):(86240, 12320, 1760, 1):336.875 KiB
// Elementwise input X_T2474 shape: fp32(1, 7, 7, 1760):(86240, 12320, 1760, 1):336.875 KiB
// Elementwise input X_I_963 shape: fp32(1760):(1):6.875 KiB
// Elementwise input X_I_962 shape: fp32(1760):(1):6.875 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2475 = add(X_T2451, X_T2474)
// Elementwise op: [[pid(Sub)]] X_T2477 = sub(X_T2475, X_I_963)
// Elementwise op: [[pid(Mul)]] X_T2478 = mul(X_T2477, X_I_962)
// Tile size: { 1, 1, 1, 1760 }
// Contraction output var shape: fp32(1, 7, 7, 1760):(86240, 12320, 1760, 1):336.875 KiB
// Computed true ops: 258720
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 880
// Computed mem write: 14080
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_860(__global float* restrict  X_T2475, __global float* restrict  X_T2478, __global const float* restrict  X_T2451, __global const float* restrict  X_T2474, __global const float* restrict  X_I_963, __global const float* restrict  X_I_962)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 6) || (i4_tid < 224));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((12320 * i2_gid) + (1760 * i3_gid)) + i4);
      float LX_T2451 = X_T2451[gout_idx];
      float LX_T2474 = X_T2474[gout_idx];
      float LX_I_963 = X_I_963[i4];
      float LX_I_962 = X_I_962[i4];
      float LX_T2475 = (LX_T2451 + LX_T2474);
      float LX_T2477 = (LX_T2475 - LX_I_963);
      float LX_T2478 = (LX_T2477 * LX_I_962);
      X_T2475[gout_idx] = LX_T2475;
      X_T2478[gout_idx] = LX_T2478;
    }
  }
}
