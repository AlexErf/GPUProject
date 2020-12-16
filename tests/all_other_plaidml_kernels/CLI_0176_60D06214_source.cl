#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i1 }
// Ranges: { 88 }
// Out stride: { 1 }
// Elementwise input X_I_449 shape: fp32(88):(1):352 bytes
// Elementwise op: [[pid(Add)]] X_T1204 = add(X_T37, X_I_449)
// Elementwise op: X_T1205 = cmp_lt(X_T1204, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T1206 = cond(X_T1205, X_T3, X_T1204)
// Elementwise op: [[pid(Sqrt)]] X_T1207 = sqrt(X_T1206)
// Tile size: { 88 }
// Contraction output var shape: fp32(88):(1):352 bytes
// Computed true ops: 352
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 12
// Computed mem write: 384
// Computed operations: 88
// Computed rollups: 0
// Computed threads used: 128
// lwork = 128, 1, 1
// gwork = 128, 1, 1
__kernel void kernel_c42_sdk_453(__global float* restrict  X_T1207, __global const float* restrict  X_I_449)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 128);
  int i1_cond = (i1_tid < 88);
  if (i1_cond)
  {
    float LX_I_449 = X_I_449[i1_tid];
    float LX_T1204 = (0.0010000000474974513f + LX_I_449);
    int LX_T1205 = (LX_T1204 < (float)0);
    float LX_T1206 = select((float)LX_T1204, (float)0, (int)LX_T1205);
    float LX_T1207 = native_sqrt(LX_T1206);
    X_T1207[i1_tid] = LX_T1207;
  }
}
