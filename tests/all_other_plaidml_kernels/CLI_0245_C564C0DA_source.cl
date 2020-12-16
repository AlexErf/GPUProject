#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 12 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 384 }
// Out stride: { 75264, 5376, 384, 1 }
// Elementwise input X_T628 shape: fp32(1, 14, 14, 384):(75264, 5376, 384, 1):294 KiB
// Elementwise input X_T651 shape: fp32(1, 14, 14, 384):(75264, 5376, 384, 1):294 KiB
// Elementwise input X_I_252 shape: fp32(384):(1):1.5 KiB
// Elementwise input X_I_251 shape: fp32(384):(1):1.5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T652 = add(X_T628, X_T651)
// Elementwise op: [[pid(Sub)]] X_T654 = sub(X_T652, X_I_252)
// Elementwise op: [[pid(Mul)]] X_T655 = mul(X_T654, X_I_251)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 384):(75264, 5376, 384, 1):294 KiB
// Computed true ops: 225792
// Computed work groups: 84
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 12, 1
__kernel void kernel_c68_sdk_215(__global float* restrict  X_T652, __global float* restrict  X_T655, __global const float* restrict  X_T628, __global const float* restrict  X_T651, __global const float* restrict  X_I_252, __global const float* restrict  X_I_251)
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
      int gout_idx = (((5376 * (i2_gid + i2_tid)) + (384 * i3)) + (i4_gid + i4_tid));
      float LX_T628 = X_T628[gout_idx];
      float LX_T651 = X_T651[gout_idx];
      float LX_I_252 = X_I_252[(i4_gid + i4_tid)];
      float LX_I_251 = X_I_251[(i4_gid + i4_tid)];
      float LX_T652 = (LX_T628 + LX_T651);
      float LX_T654 = (LX_T652 - LX_I_252);
      float LX_T655 = (LX_T654 * LX_I_251);
      X_T652[gout_idx] = LX_T652;
      X_T655[gout_idx] = LX_T655;
    }
  }
}
