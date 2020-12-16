#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3328 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1632 }
// Out stride: { 1 }
// Elementwise input X_I_844 shape: fp32(1632):(1):6.375 KiB
// Elementwise op: [[pid(Add)]] X_T2171 = add(X_T63, X_I_844)
// Elementwise op: X_T2172 = cmp_lt(X_T2171, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T2173 = cond(X_T2172, X_T36, X_T2171)
// Elementwise op: [[pid(Sqrt)]] X_T2174 = sqrt(X_T2173)
// Tile size: { 128 }
// Contraction output var shape: fp32(1632):(1):6.375 KiB
// Computed true ops: 6528
// Computed work groups: 13
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 16
// Computed mem write: 512
// Computed operations: 128
// Computed rollups: 0
// Computed threads used: 128
// lwork = 256, 1, 1
// gwork = 3328, 1, 1
__kernel void kernel_c108_sdk_754(__global float* restrict  X_T2174, __global const float* restrict  X_I_844)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 128);
  int i1_tid = (tid % 128);
  int i1_cond = ((i1_gid != 1536) || (i1_tid < 96));
  if (i1_cond)
  {
    if ((tid < 128))
    {
      int gout_idx = (i1_gid + i1_tid);
      float LX_I_844 = X_I_844[gout_idx];
      float LX_T2171 = (1.0009999641624745e-5f + LX_I_844);
      int LX_T2172 = (LX_T2171 < (float)0);
      float LX_T2173 = select((float)LX_T2171, (float)0, (int)LX_T2172);
      float LX_T2174 = native_sqrt(LX_T2173);
      X_T2174[gout_idx] = LX_T2174;
    }
  }
}
