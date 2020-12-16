#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 168 }
// Out stride: { 1 }
// Elementwise input X_I_188 shape: fp32(168):(1):672 bytes
// Elementwise op: [[pid(Add)]] X_T464 = add(X_T40, X_I_188)
// Elementwise op: X_T465 = cmp_lt(X_T464, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T466 = cond(X_T465, X_T3, X_T464)
// Elementwise op: [[pid(Sqrt)]] X_T467 = sqrt(X_T466)
// Tile size: { 168 }
// Contraction output var shape: fp32(168):(1):672 bytes
// Computed true ops: 672
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 24
// Computed mem write: 768
// Computed operations: 168
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 256, 1, 1
__kernel void kernel_c42_sdk_161(__global float* restrict  X_T467, __global const float* restrict  X_I_188)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  int i1_cond = (i1_tid < 168);
  if (i1_cond)
  {
    float LX_I_188 = X_I_188[i1_tid];
    float LX_T464 = (0.0010000000474974513f + LX_I_188);
    int LX_T465 = (LX_T464 < (float)0);
    float LX_T466 = select((float)LX_T464, (float)0, (int)LX_T465);
    float LX_T467 = native_sqrt(LX_T466);
    X_T467[i1_tid] = LX_T467;
  }
}
