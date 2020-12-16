#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 64 1 1
// lid: 64 1 1
// Names: { i1 }
// Ranges: { 44 }
// Out stride: { 1 }
// Elementwise input X_I_176 shape: fp32(44):(1):176 bytes
// Elementwise op: [[pid(Add)]] X_T463 = add(X_T37, X_I_176)
// Elementwise op: X_T464 = cmp_lt(X_T463, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T465 = cond(X_T464, X_T3, X_T463)
// Elementwise op: [[pid(Sqrt)]] X_T466 = sqrt(X_T465)
// Tile size: { 44 }
// Contraction output var shape: fp32(44):(1):176 bytes
// Computed true ops: 176
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 8
// Computed mem write: 256
// Computed operations: 44
// Computed rollups: 0
// Computed threads used: 64
// lwork = 64, 1, 1
// gwork = 64, 1, 1
__kernel void kernel_c42_sdk_161(__global float* restrict  X_T466, __global const float* restrict  X_I_176)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 64);
  int i1_cond = (i1_tid < 44);
  if (i1_cond)
  {
    float LX_I_176 = X_I_176[i1_tid];
    float LX_T463 = (0.0010000000474974513f + LX_I_176);
    int LX_T464 = (LX_T463 < (float)0);
    float LX_T465 = select((float)LX_T463, (float)0, (int)LX_T464);
    float LX_T466 = native_sqrt(LX_T465);
    X_T466[i1_tid] = LX_T466;
  }
}
