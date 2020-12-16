#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 192 }
// Out stride: { 1 }
// Elementwise input X_I_127 shape: fp32(192):(1):768 bytes
// Elementwise op: [[pid(Add)]] X_T219 = add(X_T59, X_I_127)
// Elementwise op: X_T220 = cmp_lt(X_T219, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T221 = cond(X_T220, X_T4, X_T219)
// Elementwise op: [[pid(Sqrt)]] X_T222 = sqrt(X_T221)
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
__kernel void kernel_c43_sdk_53(__global float* restrict  X_T222, __global const float* restrict  X_I_127)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  int i1_cond = (i1_tid < 192);
  if (i1_cond)
  {
    float LX_I_127 = X_I_127[i1_tid];
    float LX_T219 = (0.0010000000474974513f + LX_I_127);
    int LX_T220 = (LX_T219 < (float)0);
    float LX_T221 = select((float)LX_T219, (float)0, (int)LX_T220);
    float LX_T222 = native_sqrt(LX_T221);
    X_T222[i1_tid] = LX_T222;
  }
}
