#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 288 }
// Out stride: { 1 }
// Elementwise input X_I_142 shape: fp32(288):(1):1.125 KiB
// Elementwise op: [[pid(Add)]] X_T361 = add(X_T43, X_I_142)
// Elementwise op: X_T362 = cmp_lt(X_T361, X_T20)
// Elementwise op: [[pid(Sqrt)]] X_T363 = cond(X_T362, X_T20, X_T361)
// Elementwise op: [[pid(Sqrt)]] X_T364 = sqrt(X_T363)
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
__kernel void kernel_c68_sdk_112(__global float* restrict  X_T364, __global const float* restrict  X_I_142)
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
      float LX_T361 = (1.0009999641624745e-5f + LX_I_142);
      int LX_T362 = (LX_T361 < (float)0);
      float LX_T363 = select((float)LX_T361, (float)0, (int)LX_T362);
      float LX_T364 = native_sqrt(LX_T363);
      X_T364[gout_idx] = LX_T364;
    }
  }
}
