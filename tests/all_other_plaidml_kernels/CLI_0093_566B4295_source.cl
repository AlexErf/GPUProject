#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1536 }
// Out stride: { 1 }
// Elementwise input X_I_235 shape: fp32(1536):(1):6 KiB
// Elementwise op: [[pid(Add)]] X_T612 = add(X_T105, X_I_235)
// Elementwise op: X_T613 = cmp_lt(X_T612, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T614 = cond(X_T613, X_T3, X_T612)
// Elementwise op: [[pid(Sqrt)]] X_T615 = sqrt(X_T614)
// Tile size: { 256 }
// Contraction output var shape: fp32(1536):(1):6 KiB
// Computed true ops: 6144
// Computed work groups: 6
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 1, 1
__kernel void kernel_c28_sdk_186(__global float* restrict  X_T615, __global const float* restrict  X_I_235)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int gout_idx = (i1_gid + i1_tid);
  float LX_I_235 = X_I_235[gout_idx];
  float LX_T612 = (0.0010000000474974513f + LX_I_235);
  int LX_T613 = (LX_T612 < (float)0);
  float LX_T614 = select((float)LX_T612, (float)0, (int)LX_T613);
  float LX_T615 = native_sqrt(LX_T614);
  X_T615[gout_idx] = LX_T615;
}
