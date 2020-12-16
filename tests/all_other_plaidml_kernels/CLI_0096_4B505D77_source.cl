#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 192 }
// Out stride: { 1 }
// Elementwise input X_I_37 shape: fp32(192):(1):768 bytes
// Elementwise op: [[pid(Add)]] X_T76 = add(X_T33, X_I_37)
// Elementwise op: X_T77 = cmp_lt(X_T76, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T78 = cond(X_T77, X_T4, X_T76)
// Elementwise op: [[pid(Sqrt)]] X_T79 = sqrt(X_T78)
// Tile size: { 192 }
// Contraction output var shape: fp32(192):(1):768 bytes
// Computed true ops: 768
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 24
// Computed mem write: 768
// Computed operations: 192
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 256, 1, 1
__kernel void kernel_c51_sdk_14(__global float* restrict  X_T79, __global const float* restrict  X_I_37)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  int i1_cond = (i1_tid < 192);
  if (i1_cond)
  {
    float LX_I_37 = X_I_37[i1_tid];
    float LX_T76 = (0.0010000000474974513f + LX_I_37);
    int LX_T77 = (LX_T76 < (float)0);
    float LX_T78 = select((float)LX_T76, (float)0, (int)LX_T77);
    float LX_T79 = native_sqrt(LX_T78);
    X_T79[i1_tid] = LX_T79;
  }
}
