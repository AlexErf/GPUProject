#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 4 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 512 }
// Out stride: { 25088, 3584, 512, 1 }
// Elementwise input X_T563 shape: fp32(1, 7, 7, 512):(25088, 3584, 512, 1):98 KiB
// Elementwise input X_T567 shape: fp32(512):(1):2 KiB
// Elementwise input X_I_42 shape: fp32(512):(1):2 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T568 = div(X_T563, X_T567)
// Elementwise op: [[pid(Add, Switch)]] X_T569 = add(X_T568, X_I_42)
// Elementwise op: X_T570 = cmp_lt(X_T569, X_T2)
// Elementwise op: [[pid(Relu)]] X_T571 = cond(X_T570, X_T2, X_T569)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 512):(25088, 3584, 512, 1):98 KiB
// Computed true ops: 100352
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
__kernel void kernel_c29_sdk_134(__global float* restrict  X_T571, __global const float* restrict  X_T563, __global const float* restrict  X_T567, __global const float* restrict  X_I_42)
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
      float LX_T563 = X_T563[gout_idx];
      float LX_T567 = X_T567[(i4_gid + i4)];
      float LX_I_42 = X_I_42[(i4_gid + i4)];
      float LX_T568 = (LX_T563 / LX_T567);
      float LX_T569 = (LX_T568 + LX_I_42);
      int LX_T570 = (LX_T569 < 0.0f);
      float LX_T571 = select((float)LX_T569, (float)0.0f, (int)LX_T570);
      X_T571[gout_idx] = LX_T571;
    }
  }
}
