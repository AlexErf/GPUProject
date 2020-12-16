#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 512 }
// Out stride: { 401408, 14336, 512, 1 }
// Elementwise input X_T244 shape: fp32(1, 28, 28, 512):(401408, 14336, 512, 1):1568 KiB
// Elementwise input X_T248 shape: fp32(512):(1):2 KiB
// Elementwise input X_I_167 shape: fp32(512):(1):2 KiB
// Elementwise input X_T216 shape: fp32(1, 28, 28, 512):(401408, 14336, 512, 1):1568 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T249 = div(X_T244, X_T248)
// Elementwise op: [[pid(Add, Switch)]] X_T250 = add(X_T249, X_I_167)
// Elementwise op: [[pid(Add)]] X_T251 = add(X_T250, X_T216)
// Elementwise op: X_T252 = cmp_lt(X_T251, X_T2)
// Elementwise op: [[pid(Relu)]] X_T253 = cond(X_T252, X_T2, X_T251)
// Tile size: { 1, 28, 2, 64 }
// Contraction output var shape: fp32(1, 28, 28, 512):(401408, 14336, 512, 1):1568 KiB
// Computed true ops: 2007040
// Computed work groups: 112
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1792
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 8, 1
__kernel void kernel_c29_sdk_56(__global float* restrict  X_T253, __global const float* restrict  X_T244, __global const float* restrict  X_T248, __global const float* restrict  X_I_167, __global const float* restrict  X_T216)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 64);
  int i3_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 2);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
  {
    int i4 = ((32 * i4_lid) + i4_tid);
    for (int i2_lid = 0; i2_lid < 7; i2_lid += 1)
    {
      int i2 = ((4 * i2_lid) + i2_tid);
      int gout_idx = (((14336 * i2) + (512 * (i3_gid + i3_tid))) + (i4_gid + i4));
      float LX_T244 = X_T244[gout_idx];
      float LX_T248 = X_T248[(i4_gid + i4)];
      float LX_I_167 = X_I_167[(i4_gid + i4)];
      float LX_T216 = X_T216[gout_idx];
      float LX_T249 = (LX_T244 / LX_T248);
      float LX_T250 = (LX_T249 + LX_I_167);
      float LX_T251 = (LX_T250 + LX_T216);
      int LX_T252 = (LX_T251 < 0.0f);
      float LX_T253 = select((float)LX_T251, (float)0.0f, (int)LX_T252);
      X_T253[gout_idx] = LX_T253;
    }
  }
}
