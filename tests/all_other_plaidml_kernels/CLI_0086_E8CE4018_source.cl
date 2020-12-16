#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1024 }
// Out stride: { 1 }
// Elementwise input X_I_229 shape: fp32(1024):(1):4 KiB
// Elementwise op: [[pid(Add)]] X_T586 = add(X_T105, X_I_229)
// Elementwise op: X_T587 = cmp_lt(X_T586, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T588 = cond(X_T587, X_T3, X_T586)
// Elementwise op: [[pid(Sqrt)]] X_T589 = sqrt(X_T588)
// Tile size: { 256 }
// Contraction output var shape: fp32(1024):(1):4 KiB
// Computed true ops: 4096
// Computed work groups: 4
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1024, 1, 1
__kernel void kernel_c28_sdk_178(__global float* restrict  X_T589, __global const float* restrict  X_I_229)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int gout_idx = (i1_gid + i1_tid);
  float LX_I_229 = X_I_229[gout_idx];
  float LX_T586 = (0.0010000000474974513f + LX_I_229);
  int LX_T587 = (LX_T586 < (float)0);
  float LX_T588 = select((float)LX_T586, (float)0, (int)LX_T587);
  float LX_T589 = native_sqrt(LX_T588);
  X_T589[gout_idx] = LX_T589;
}
