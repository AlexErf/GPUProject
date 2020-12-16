#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1152 }
// Out stride: { 1 }
// Elementwise input X_I_493 shape: fp32(1152):(1):4.5 KiB
// Elementwise op: [[pid(Add)]] X_T1284 = add(X_T71, X_I_493)
// Elementwise op: X_T1285 = cmp_lt(X_T1284, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1286 = cond(X_T1285, X_T36, X_T1284)
// Elementwise op: [[pid(Sqrt)]] X_T1287 = sqrt(X_T1286)
// Tile size: { 256 }
// Contraction output var shape: fp32(1152):(1):4.5 KiB
// Computed true ops: 4608
// Computed work groups: 5
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1280, 1, 1
__kernel void kernel_c124_sdk_433(__global float* restrict  X_T1287, __global const float* restrict  X_I_493)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1024) || (i1_tid < 128));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_493 = X_I_493[gout_idx];
    float LX_T1284 = (1.0009999641624745e-5f + LX_I_493);
    int LX_T1285 = (LX_T1284 < (float)0);
    float LX_T1286 = select((float)LX_T1284, (float)0, (int)LX_T1285);
    float LX_T1287 = native_sqrt(LX_T1286);
    X_T1287[gout_idx] = LX_T1287;
  }
}
