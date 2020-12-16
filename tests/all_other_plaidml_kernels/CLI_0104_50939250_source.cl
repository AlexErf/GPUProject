#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 64 1 1
// lid: 64 1 1
// Names: { i1 }
// Ranges: { 48 }
// Out stride: { 1 }
// Elementwise input X_I_45 shape: fp32(48):(1):192 bytes
// Elementwise op: [[pid(Add)]] X_T103 = add(X_T33, X_I_45)
// Elementwise op: X_T104 = cmp_lt(X_T103, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T105 = cond(X_T104, X_T4, X_T103)
// Elementwise op: [[pid(Sqrt)]] X_T106 = sqrt(X_T105)
// Tile size: { 48 }
// Contraction output var shape: fp32(48):(1):192 bytes
// Computed true ops: 192
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 8
// Computed mem write: 256
// Computed operations: 48
// Computed rollups: 0
// Computed threads used: 64
// lwork = 64, 1, 1
// gwork = 64, 1, 1
__kernel void kernel_c51_sdk_22(__global float* restrict  X_T106, __global const float* restrict  X_I_45)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 64);
  int i1_cond = (i1_tid < 48);
  if (i1_cond)
  {
    float LX_I_45 = X_I_45[i1_tid];
    float LX_T103 = (0.0010000000474974513f + LX_I_45);
    int LX_T104 = (LX_T103 < (float)0);
    float LX_T105 = select((float)LX_T103, (float)0, (int)LX_T104);
    float LX_T106 = native_sqrt(LX_T105);
    X_T106[i1_tid] = LX_T106;
  }
}
