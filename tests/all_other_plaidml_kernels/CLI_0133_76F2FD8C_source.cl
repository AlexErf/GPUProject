#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 256 }
// Out stride: { 1 }
// Elementwise input X_I_333 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(Add)]] X_T900 = add(X_T33, X_I_333)
// Elementwise op: X_T901 = cmp_lt(X_T900, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T902 = cond(X_T901, X_T4, X_T900)
// Elementwise op: [[pid(Sqrt)]] X_T903 = sqrt(X_T902)
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
__kernel void kernel_c51_sdk_293(__global float* restrict  X_T903, __global const float* restrict  X_I_333)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  float LX_I_333 = X_I_333[i1_tid];
  float LX_T900 = (0.0010000000474974513f + LX_I_333);
  int LX_T901 = (LX_T900 < (float)0);
  float LX_T902 = select((float)LX_T900, (float)0, (int)LX_T901);
  float LX_T903 = native_sqrt(LX_T902);
  X_T903[i1_tid] = LX_T903;
}
