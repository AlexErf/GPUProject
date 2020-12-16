#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i1 }
// Ranges: { 96 }
// Out stride: { 1 }
// Elementwise input X_I_41 shape: fp32(96):(1):384 bytes
// Elementwise op: [[pid(Add)]] X_T118 = add(X_T71, X_I_41)
// Elementwise op: X_T119 = cmp_lt(X_T118, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T120 = cond(X_T119, X_T36, X_T118)
// Elementwise op: [[pid(Sqrt)]] X_T121 = sqrt(X_T120)
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
__kernel void kernel_c124_sdk_16(__global float* restrict  X_T121, __global const float* restrict  X_I_41)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 128);
  int i1_cond = (i1_tid < 96);
  if (i1_cond)
  {
    float LX_I_41 = X_I_41[i1_tid];
    float LX_T118 = (1.0009999641624745e-5f + LX_I_41);
    int LX_T119 = (LX_T118 < (float)0);
    float LX_T120 = select((float)LX_T118, (float)0, (int)LX_T119);
    float LX_T121 = native_sqrt(LX_T120);
    X_T121[i1_tid] = LX_T121;
  }
}
