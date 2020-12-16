#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 288 }
// Out stride: { 1 }
// Elementwise input X_I_142 shape: fp32(288):(1):1.125 KiB
// Elementwise op: [[pid(Add)]] X_T381 = add(X_T63, X_I_142)
// Elementwise op: X_T382 = cmp_lt(X_T381, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T383 = cond(X_T382, X_T36, X_T381)
// Elementwise op: [[pid(Sqrt)]] X_T384 = sqrt(X_T383)
// Tile size: { 128 }
// Contraction output var shape: fp32(288):(1):1.125 KiB
// Computed true ops: 1152
// Computed work groups: 3
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 16
// Computed mem write: 512
// Computed operations: 128
// Computed rollups: 0
// Computed threads used: 128
// lwork = 256, 1, 1
// gwork = 768, 1, 1
__kernel void kernel_c108_sdk_112(__global float* restrict  X_T384, __global const float* restrict  X_I_142)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 128);
  int i1_tid = (tid % 128);
  int i1_cond = ((i1_gid != 256) || (i1_tid < 32));
  if (i1_cond)
  {
    if ((tid < 128))
    {
      int gout_idx = (i1_gid + i1_tid);
      float LX_I_142 = X_I_142[gout_idx];
      float LX_T381 = (1.0009999641624745e-5f + LX_I_142);
      int LX_T382 = (LX_T381 < (float)0);
      float LX_T383 = select((float)LX_T381, (float)0, (int)LX_T382);
      float LX_T384 = native_sqrt(LX_T383);
      X_T384[gout_idx] = LX_T384;
    }
  }
}
