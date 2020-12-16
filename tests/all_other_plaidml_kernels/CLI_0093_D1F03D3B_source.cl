#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i1 }
// Ranges: { 80 }
// Out stride: { 1 }
// Elementwise input X_I_36 shape: fp32(80):(1):320 bytes
// Elementwise op: [[pid(Add)]] X_T65 = add(X_T33, X_I_36)
// Elementwise op: X_T66 = cmp_lt(X_T65, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T67 = cond(X_T66, X_T4, X_T65)
// Elementwise op: [[pid(Sqrt)]] X_T68 = sqrt(X_T67)
// Tile size: { 80 }
// Contraction output var shape: fp32(80):(1):320 bytes
// Computed true ops: 320
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 12
// Computed mem write: 384
// Computed operations: 80
// Computed rollups: 0
// Computed threads used: 128
// lwork = 128, 1, 1
// gwork = 128, 1, 1
__kernel void kernel_c51_sdk_11(__global float* restrict  X_T68, __global const float* restrict  X_I_36)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 128);
  int i1_cond = (i1_tid < 80);
  if (i1_cond)
  {
    float LX_I_36 = X_I_36[i1_tid];
    float LX_T65 = (0.0010000000474974513f + LX_I_36);
    int LX_T66 = (LX_T65 < (float)0);
    float LX_T67 = select((float)LX_T65, (float)0, (int)LX_T66);
    float LX_T68 = native_sqrt(LX_T67);
    X_T68[i1_tid] = LX_T68;
  }
}
