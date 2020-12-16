#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1184 }
// Out stride: { 1 }
// Elementwise input X_I_503 shape: fp32(1184):(1):4.625 KiB
// Elementwise op: [[pid(Add)]] X_T1301 = add(X_T63, X_I_503)
// Elementwise op: X_T1302 = cmp_lt(X_T1301, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1303 = cond(X_T1302, X_T36, X_T1301)
// Elementwise op: [[pid(Sqrt)]] X_T1304 = sqrt(X_T1303)
// Tile size: { 256 }
// Contraction output var shape: fp32(1184):(1):4.625 KiB
// Computed true ops: 4736
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
__kernel void kernel_c108_sdk_442(__global float* restrict  X_T1304, __global const float* restrict  X_I_503)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1024) || (i1_tid < 160));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_503 = X_I_503[gout_idx];
    float LX_T1301 = (1.0009999641624745e-5f + LX_I_503);
    int LX_T1302 = (LX_T1301 < (float)0);
    float LX_T1303 = select((float)LX_T1301, (float)0, (int)LX_T1302);
    float LX_T1304 = native_sqrt(LX_T1303);
    X_T1304[gout_idx] = LX_T1304;
  }
}
