#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 14 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 224 }
// Out stride: { 702464, 12544, 224, 1 }
// Elementwise input X_T217 shape: fp32(1, 56, 56, 224):(702464, 12544, 224, 1):2744 KiB
// Elementwise input X_T221 shape: fp32(224):(1):896 bytes
// Elementwise input X_I_78 shape: fp32(224):(1):896 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T222 = div(X_T217, X_T221)
// Elementwise op: [[pid(Add, Switch)]] X_T223 = add(X_T222, X_I_78)
// Elementwise op: X_T224 = cmp_lt(X_T223, X_T2)
// Elementwise op: [[pid(Relu)]] X_T225 = cond(X_T224, X_T2, X_T223)
// Tile size: { 1, 4, 4, 224 }
// Contraction output var shape: fp32(1, 56, 56, 224):(702464, 12544, 224, 1):2744 KiB
// Computed true ops: 2809856
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1344
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 14, 1
__kernel void kernel_c124_sdk_53(__global float* restrict  X_T225, __global const float* restrict  X_T217, __global const float* restrict  X_T221, __global const float* restrict  X_I_78)
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
      float LX_T217 = X_T217[gout_idx];
      float LX_T221 = X_T221[i4];
      float LX_I_78 = X_I_78[i4];
      float LX_T222 = (LX_T217 / LX_T221);
      float LX_T223 = (LX_T222 + LX_I_78);
      int LX_T224 = (LX_T223 < 0.0f);
      float LX_T225 = select((float)LX_T223, (float)0.0f, (int)LX_T224);
      X_T225[gout_idx] = LX_T225;
    }
  }
}
