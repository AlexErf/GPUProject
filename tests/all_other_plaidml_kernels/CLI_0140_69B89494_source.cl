#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 960 }
// Out stride: { 1 }
// Elementwise input X_I_241 shape: fp32(960):(1):3.75 KiB
// Elementwise op: [[pid(Add)]] X_T597 = add(X_T59, X_I_241)
// Elementwise op: X_T598 = cmp_lt(X_T597, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T599 = cond(X_T598, X_T4, X_T597)
// Elementwise op: [[pid(Sqrt)]] X_T600 = sqrt(X_T599)
// Tile size: { 256 }
// Contraction output var shape: fp32(960):(1):3.75 KiB
// Computed true ops: 3840
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
__kernel void kernel_c43_sdk_162(__global float* restrict  X_T600, __global const float* restrict  X_I_241)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 768) || (i1_tid < 192));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_241 = X_I_241[gout_idx];
    float LX_T597 = (0.0010000000474974513f + LX_I_241);
    int LX_T598 = (LX_T597 < (float)0);
    float LX_T599 = select((float)LX_T597, (float)0, (int)LX_T598);
    float LX_T600 = native_sqrt(LX_T599);
    X_T600[gout_idx] = LX_T600;
  }
}
