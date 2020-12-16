#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 256 }
// Out stride: { 1 }
// Elementwise input X_I_190 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(Add)]] X_T171 = add(X_T105, X_I_190)
// Elementwise op: X_T172 = cmp_lt(X_T171, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T173 = cond(X_T172, X_T3, X_T171)
// Elementwise op: [[pid(Sqrt)]] X_T174 = sqrt(X_T173)
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
__kernel void kernel_c28_sdk_54(__global float* restrict  X_T174, __global const float* restrict  X_I_190)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  float LX_I_190 = X_I_190[i1_tid];
  float LX_T171 = (0.0010000000474974513f + LX_I_190);
  int LX_T172 = (LX_T171 < (float)0);
  float LX_T173 = select((float)LX_T171, (float)0, (int)LX_T172);
  float LX_T174 = native_sqrt(LX_T173);
  X_T174[i1_tid] = LX_T174;
}
