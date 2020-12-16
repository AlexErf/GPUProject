#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 176 }
// Out stride: { 1 }
// Elementwise input X_I_788 shape: fp32(176):(1):704 bytes
// Elementwise op: [[pid(Add)]] X_T2132 = add(X_T37, X_I_788)
// Elementwise op: X_T2133 = cmp_lt(X_T2132, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T2134 = cond(X_T2133, X_T3, X_T2132)
// Elementwise op: [[pid(Sqrt)]] X_T2135 = sqrt(X_T2134)
// Tile size: { 176 }
// Contraction output var shape: fp32(176):(1):704 bytes
// Computed true ops: 704
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 24
// Computed mem write: 768
// Computed operations: 176
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 256, 1, 1
__kernel void kernel_c42_sdk_817(__global float* restrict  X_T2135, __global const float* restrict  X_I_788)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  int i1_cond = (i1_tid < 176);
  if (i1_cond)
  {
    float LX_I_788 = X_I_788[i1_tid];
    float LX_T2132 = (0.0010000000474974513f + LX_I_788);
    int LX_T2133 = (LX_T2132 < (float)0);
    float LX_T2134 = select((float)LX_T2132, (float)0, (int)LX_T2133);
    float LX_T2135 = native_sqrt(LX_T2134);
    X_T2135[i1_tid] = LX_T2135;
  }
}
