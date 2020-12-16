#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 160 }
// Out stride: { 1 }
// Elementwise input X_I_61 shape: fp32(160):(1):640 bytes
// Elementwise op: [[pid(Add)]] X_T160 = add(X_T63, X_I_61)
// Elementwise op: X_T161 = cmp_lt(X_T160, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T162 = cond(X_T161, X_T36, X_T160)
// Elementwise op: [[pid(Sqrt)]] X_T163 = sqrt(X_T162)
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
__kernel void kernel_c108_sdk_34(__global float* restrict  X_T163, __global const float* restrict  X_I_61)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  int i1_cond = (i1_tid < 160);
  if (i1_cond)
  {
    float LX_I_61 = X_I_61[i1_tid];
    float LX_T160 = (1.0009999641624745e-5f + LX_I_61);
    int LX_T161 = (LX_T160 < (float)0);
    float LX_T162 = select((float)LX_T160, (float)0, (int)LX_T161);
    float LX_T163 = native_sqrt(LX_T162);
    X_T163[i1_tid] = LX_T163;
  }
}
