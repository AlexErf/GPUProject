#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 728 }
// Out stride: { 1 }
// Elementwise input X_I_197 shape: fp32(728):(1):2.84375 KiB
// Elementwise op: [[pid(Add)]] X_T212 = add(X_T105, X_I_197)
// Elementwise op: X_T213 = cmp_lt(X_T212, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T214 = cond(X_T213, X_T3, X_T212)
// Elementwise op: [[pid(Sqrt)]] X_T215 = sqrt(X_T214)
// Tile size: { 256 }
// Contraction output var shape: fp32(728):(1):2.84375 KiB
// Computed true ops: 2912
// Computed work groups: 3
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 1, 1
__kernel void kernel_c28_sdk_66(__global float* restrict  X_T215, __global const float* restrict  X_I_197)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 512) || (i1_tid < 216));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_197 = X_I_197[gout_idx];
    float LX_T212 = (0.0010000000474974513f + LX_I_197);
    int LX_T213 = (LX_T212 < (float)0);
    float LX_T214 = select((float)LX_T212, (float)0, (int)LX_T213);
    float LX_T215 = native_sqrt(LX_T214);
    X_T215[gout_idx] = LX_T215;
  }
}
