#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 160 }
// Out stride: { 1 }
// Elementwise input X_I_352 shape: fp32(160):(1):640 bytes
// Elementwise op: [[pid(Add)]] X_T962 = add(X_T33, X_I_352)
// Elementwise op: X_T963 = cmp_lt(X_T962, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T964 = cond(X_T963, X_T4, X_T962)
// Elementwise op: [[pid(Sqrt)]] X_T965 = sqrt(X_T964)
// Tile size: { 160 }
// Contraction output var shape: fp32(160):(1):640 bytes
// Computed true ops: 640
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 20
// Computed mem write: 640
// Computed operations: 160
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 256, 1, 1
__kernel void kernel_c51_sdk_314(__global float* restrict  X_T965, __global const float* restrict  X_I_352)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  int i1_cond = (i1_tid < 160);
  if (i1_cond)
  {
    float LX_I_352 = X_I_352[i1_tid];
    float LX_T962 = (0.0010000000474974513f + LX_I_352);
    int LX_T963 = (LX_T962 < (float)0);
    float LX_T964 = select((float)LX_T962, (float)0, (int)LX_T963);
    float LX_T965 = native_sqrt(LX_T964);
    X_T965[i1_tid] = LX_T965;
  }
}
