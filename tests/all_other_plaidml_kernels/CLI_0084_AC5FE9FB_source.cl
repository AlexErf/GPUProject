#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 4 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 512 }
// Out stride: { 25088, 3584, 512, 1 }
// Elementwise input X_T395 shape: fp32(1, 7, 7, 512):(25088, 3584, 512, 1):98 KiB
// Elementwise input X_T399 shape: fp32(512):(1):2 KiB
// Elementwise input X_I_15 shape: fp32(512):(1):2 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T400 = div(X_T395, X_T399)
// Elementwise op: [[pid(Add, Switch)]] X_T401 = add(X_T400, X_I_15)
// Elementwise op: X_T402 = cmp_lt(X_T401, X_T10)
// Elementwise op: [[pid(Relu)]] X_T403 = cond(X_T402, X_T10, X_T401)
// Elementwise op: X_T404 = cmp_lt(X_T403, X_T9)
// Elementwise op: [[pid(Relu)]] X_T405 = cond(X_T404, X_T403, X_T9)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 512):(25088, 3584, 512, 1):98 KiB
// Computed true ops: 150528
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 4, 1
__kernel void kernel_c25_sdk_102(__global float* restrict  X_T405, __global const float* restrict  X_T395, __global const float* restrict  X_T399, __global const float* restrict  X_I_15)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  int i3_cond = (i3_tid < 7);
  if (i3_cond)
  {
    for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int gout_idx = (((3584 * i2_gid) + (512 * i3_tid)) + (i4_gid + i4));
      float LX_T395 = X_T395[gout_idx];
      float LX_T399 = X_T399[(i4_gid + i4)];
      float LX_I_15 = X_I_15[(i4_gid + i4)];
      float LX_T400 = (LX_T395 / LX_T399);
      float LX_T401 = (LX_T400 + LX_I_15);
      int LX_T402 = (LX_T401 < 0.0f);
      float LX_T403 = select((float)LX_T401, (float)0.0f, (int)LX_T402);
      int LX_T404 = (LX_T403 < 6.0f);
      float LX_T405 = select((float)6.0f, (float)LX_T403, (int)LX_T404);
      X_T405[gout_idx] = LX_T405;
    }
  }
}
