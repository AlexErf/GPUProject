#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 160 }
// Out stride: { 1 }
// Elementwise input X_I_61 shape: fp32(160):(1):640 bytes
// Elementwise op: [[pid(Add)]] X_T168 = add(X_T71, X_I_61)
// Elementwise op: X_T169 = cmp_lt(X_T168, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T170 = cond(X_T169, X_T36, X_T168)
// Elementwise op: [[pid(Sqrt)]] X_T171 = sqrt(X_T170)
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
__kernel void kernel_c124_sdk_34(__global float* restrict  X_T171, __global const float* restrict  X_I_61)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  int i1_cond = (i1_tid < 160);
  if (i1_cond)
  {
    float LX_I_61 = X_I_61[i1_tid];
    float LX_T168 = (1.0009999641624745e-5f + LX_I_61);
    int LX_T169 = (LX_T168 < (float)0);
    float LX_T170 = select((float)LX_T168, (float)0, (int)LX_T169);
    float LX_T171 = native_sqrt(LX_T170);
    X_T171[i1_tid] = LX_T171;
  }
}
