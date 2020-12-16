#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1120 }
// Out stride: { 1 }
// Elementwise input X_I_483 shape: fp32(1120):(1):4.375 KiB
// Elementwise op: [[pid(Add)]] X_T1251 = add(X_T63, X_I_483)
// Elementwise op: X_T1252 = cmp_lt(X_T1251, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1253 = cond(X_T1252, X_T36, X_T1251)
// Elementwise op: [[pid(Sqrt)]] X_T1254 = sqrt(X_T1253)
// Tile size: { 256 }
// Contraction output var shape: fp32(1120):(1):4.375 KiB
// Computed true ops: 4480
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
__kernel void kernel_c108_sdk_424(__global float* restrict  X_T1254, __global const float* restrict  X_I_483)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1024) || (i1_tid < 96));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_483 = X_I_483[gout_idx];
    float LX_T1251 = (1.0009999641624745e-5f + LX_I_483);
    int LX_T1252 = (LX_T1251 < (float)0);
    float LX_T1253 = select((float)LX_T1251, (float)0, (int)LX_T1252);
    float LX_T1254 = native_sqrt(LX_T1253);
    X_T1254[gout_idx] = LX_T1254;
  }
}
