#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 32 1 1
// lid: 32 1 1
// Names: { i1 }
// Ranges: { 22 }
// Out stride: { 1 }
// Elementwise input X_I_109 shape: fp32(22):(1):88 bytes
// Elementwise op: [[pid(Add)]] X_T241 = add(X_T37, X_I_109)
// Elementwise op: X_T242 = cmp_lt(X_T241, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T243 = cond(X_T242, X_T3, X_T241)
// Elementwise op: [[pid(Sqrt)]] X_T244 = sqrt(X_T243)
// Tile size: { 22 }
// Contraction output var shape: fp32(22):(1):88 bytes
// Computed true ops: 88
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 4
// Computed mem write: 128
// Computed operations: 22
// Computed rollups: 0
// Computed threads used: 32
// lwork = 32, 1, 1
// gwork = 32, 1, 1
__kernel void kernel_c42_sdk_75(__global float* restrict  X_T244, __global const float* restrict  X_I_109)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 32);
  int i1_cond = (i1_tid < 22);
  if (i1_cond)
  {
    float LX_I_109 = X_I_109[i1_tid];
    float LX_T241 = (0.0010000000474974513f + LX_I_109);
    int LX_T242 = (LX_T241 < (float)0);
    float LX_T243 = select((float)LX_T241, (float)0, (int)LX_T242);
    float LX_T244 = native_sqrt(LX_T243);
    X_T244[i1_tid] = LX_T244;
  }
}
