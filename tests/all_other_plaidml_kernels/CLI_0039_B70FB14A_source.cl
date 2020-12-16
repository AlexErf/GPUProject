#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 256 }
// Out stride: { 1 }
// Elementwise input X_I_251 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(Add)]] X_T77 = add(X_T37, X_I_251)
// Elementwise op: X_T78 = cmp_lt(X_T77, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T79 = cond(X_T78, X_T3, X_T77)
// Elementwise op: [[pid(Sqrt)]] X_T80 = sqrt(X_T79)
// Tile size: { 256 }
// Contraction output var shape: fp32(256):(1):1 KiB
// Computed true ops: 1024
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 256, 1, 1
__kernel void kernel_c29_sdk_13(__global float* restrict  X_T80, __global const float* restrict  X_I_251)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  float LX_I_251 = X_I_251[i1_tid];
  float LX_T77 = (0.0010000000474974513f + LX_I_251);
  int LX_T78 = (LX_T77 < (float)0);
  float LX_T79 = select((float)LX_T77, (float)0, (int)LX_T78);
  float LX_T80 = native_sqrt(LX_T79);
  X_T80[i1_tid] = LX_T80;
}
