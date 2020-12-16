#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 864 }
// Out stride: { 1 }
// Elementwise input X_I_403 shape: fp32(864):(1):3.375 KiB
// Elementwise op: [[pid(Add)]] X_T1059 = add(X_T71, X_I_403)
// Elementwise op: X_T1060 = cmp_lt(X_T1059, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1061 = cond(X_T1060, X_T36, X_T1059)
// Elementwise op: [[pid(Sqrt)]] X_T1062 = sqrt(X_T1061)
// Tile size: { 256 }
// Contraction output var shape: fp32(864):(1):3.375 KiB
// Computed true ops: 3456
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
__kernel void kernel_c124_sdk_352(__global float* restrict  X_T1062, __global const float* restrict  X_I_403)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 768) || (i1_tid < 96));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_403 = X_I_403[gout_idx];
    float LX_T1059 = (1.0009999641624745e-5f + LX_I_403);
    int LX_T1060 = (LX_T1059 < (float)0);
    float LX_T1061 = select((float)LX_T1059, (float)0, (int)LX_T1060);
    float LX_T1062 = native_sqrt(LX_T1061);
    X_T1062[gout_idx] = LX_T1062;
  }
}
