#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 14 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 224 }
// Out stride: { 702464, 12544, 224, 1 }
// Elementwise input X_T182 shape: fp32(1, 56, 56, 224):(702464, 12544, 224, 1):2744 KiB
// Elementwise input X_T205 shape: fp32(1, 56, 56, 224):(702464, 12544, 224, 1):2744 KiB
// Elementwise input X_I_80 shape: fp32(224):(1):896 bytes
// Elementwise input X_I_79 shape: fp32(224):(1):896 bytes
// Elementwise op: [[pid(Concatenate)]] X_T206 = add(X_T182, X_T205)
// Elementwise op: [[pid(Sub)]] X_T208 = sub(X_T206, X_I_80)
// Elementwise op: [[pid(Mul)]] X_T209 = mul(X_T208, X_I_79)
// Tile size: { 1, 4, 4, 224 }
// Contraction output var shape: fp32(1, 56, 56, 224):(702464, 12544, 224, 1):2744 KiB
// Computed true ops: 2107392
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1792
// Computed mem write: 28672
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 14, 1
__kernel void kernel_c108_sdk_50(__global float* restrict  X_T206, __global float* restrict  X_T209, __global const float* restrict  X_T182, __global const float* restrict  X_T205, __global const float* restrict  X_I_80, __global const float* restrict  X_I_79)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 4);
  int i2_gid = (get_group_id(1) * 4);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4 = ((32 * i4_lid) + i4_tid);
    for (int i2_lid = 0; i2_lid < 2; i2_lid += 1)
    {
      int i2 = ((2 * i2_lid) + i2_tid);
      int gout_idx = (((12544 * (i2_gid + i2)) + (224 * (i3_gid + i3_tid))) + i4);
      float LX_T182 = X_T182[gout_idx];
      float LX_T205 = X_T205[gout_idx];
      float LX_I_80 = X_I_80[i4];
      float LX_I_79 = X_I_79[i4];
      float LX_T206 = (LX_T182 + LX_T205);
      float LX_T208 = (LX_T206 - LX_I_80);
      float LX_T209 = (LX_T208 * LX_I_79);
      X_T206[gout_idx] = LX_T206;
      X_T209[gout_idx] = LX_T209;
    }
  }
}
