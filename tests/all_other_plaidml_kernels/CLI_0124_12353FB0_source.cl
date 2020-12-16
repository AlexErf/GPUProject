#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i1 }
// Ranges: { 96 }
// Out stride: { 1 }
// Elementwise input X_I_78 shape: fp32(96):(1):384 bytes
// Elementwise op: [[pid(Add)]] X_T139 = add(X_T33, X_I_78)
// Elementwise op: X_T140 = cmp_lt(X_T139, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T141 = cond(X_T140, X_T4, X_T139)
// Elementwise op: [[pid(Sqrt)]] X_T142 = sqrt(X_T141)
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
__kernel void kernel_c56_sdk_33(__global float* restrict  X_T142, __global const float* restrict  X_I_78)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 128);
  int i1_cond = (i1_tid < 96);
  if (i1_cond)
  {
    float LX_I_78 = X_I_78[i1_tid];
    float LX_T139 = (0.0010000000474974513f + LX_I_78);
    int LX_T140 = (LX_T139 < (float)0);
    float LX_T141 = select((float)LX_T139, (float)0, (int)LX_T140);
    float LX_T142 = native_sqrt(LX_T141);
    X_T142[i1_tid] = LX_T142;
  }
}
