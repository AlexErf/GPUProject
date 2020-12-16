#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 14 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 224 }
// Out stride: { 702464, 12544, 224, 1 }
// Elementwise input X_T209 shape: fp32(1, 56, 56, 224):(702464, 12544, 224, 1):2744 KiB
// Elementwise input X_T213 shape: fp32(224):(1):896 bytes
// Elementwise input X_I_78 shape: fp32(224):(1):896 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T214 = div(X_T209, X_T213)
// Elementwise op: [[pid(Add, Switch)]] X_T215 = add(X_T214, X_I_78)
// Elementwise op: X_T216 = cmp_lt(X_T215, X_T2)
// Elementwise op: [[pid(Relu)]] X_T217 = cond(X_T216, X_T2, X_T215)
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
__kernel void kernel_c108_sdk_53(__global float* restrict  X_T217, __global const float* restrict  X_T209, __global const float* restrict  X_T213, __global const float* restrict  X_I_78)
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
      float LX_T209 = X_T209[gout_idx];
      float LX_T213 = X_T213[i4];
      float LX_I_78 = X_I_78[i4];
      float LX_T214 = (LX_T209 / LX_T213);
      float LX_T215 = (LX_T214 + LX_I_78);
      int LX_T216 = (LX_T215 < 0.0f);
      float LX_T217 = select((float)LX_T215, (float)0.0f, (int)LX_T216);
      X_T217[gout_idx] = LX_T217;
    }
  }
}
