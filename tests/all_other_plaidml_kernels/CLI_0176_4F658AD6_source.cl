#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 160 }
// Out stride: { 1 }
// Elementwise input X_I_194 shape: fp32(160):(1):640 bytes
// Elementwise op: [[pid(Add)]] X_T517 = add(X_T33, X_I_194)
// Elementwise op: X_T518 = cmp_lt(X_T517, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T519 = cond(X_T518, X_T4, X_T517)
// Elementwise op: [[pid(Sqrt)]] X_T520 = sqrt(X_T519)
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
__kernel void kernel_c56_sdk_173(__global float* restrict  X_T520, __global const float* restrict  X_I_194)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  int i1_cond = (i1_tid < 160);
  if (i1_cond)
  {
    float LX_I_194 = X_I_194[i1_tid];
    float LX_T517 = (0.0010000000474974513f + LX_I_194);
    int LX_T518 = (LX_T517 < (float)0);
    float LX_T519 = select((float)LX_T517, (float)0, (int)LX_T518);
    float LX_T520 = native_sqrt(LX_T519);
    X_T520[i1_tid] = LX_T520;
  }
}
