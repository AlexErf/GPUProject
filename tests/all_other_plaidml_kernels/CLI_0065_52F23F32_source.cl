#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i1 }
// Ranges: { 96 }
// Out stride: { 1 }
// Elementwise input X_I_59 shape: fp32(96):(1):384 bytes
// Elementwise op: [[pid(Add)]] X_T41 = add(X_T40, X_I_59)
// Elementwise op: X_T42 = cmp_lt(X_T41, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T43 = cond(X_T42, X_T3, X_T41)
// Elementwise op: [[pid(Sqrt)]] X_T44 = sqrt(X_T43)
// Tile size: { 96 }
// Contraction output var shape: fp32(96):(1):384 bytes
// Computed true ops: 384
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 12
// Computed mem write: 384
// Computed operations: 96
// Computed rollups: 0
// Computed threads used: 128
// lwork = 128, 1, 1
// gwork = 128, 1, 1
__kernel void kernel_c42_sdk_1(__global float* restrict  X_T44, __global const float* restrict  X_I_59)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 128);
  int i1_cond = (i1_tid < 96);
  if (i1_cond)
  {
    float LX_I_59 = X_I_59[i1_tid];
    float LX_T41 = (0.0010000000474974513f + LX_I_59);
    int LX_T42 = (LX_T41 < (float)0);
    float LX_T43 = select((float)LX_T41, (float)0, (int)LX_T42);
    float LX_T44 = native_sqrt(LX_T43);
    X_T44[i1_tid] = LX_T44;
  }
}
