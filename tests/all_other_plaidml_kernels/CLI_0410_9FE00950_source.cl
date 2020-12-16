#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 960 }
// Out stride: { 1 }
// Elementwise input X_I_433 shape: fp32(960):(1):3.75 KiB
// Elementwise op: [[pid(Add)]] X_T1126 = add(X_T63, X_I_433)
// Elementwise op: X_T1127 = cmp_lt(X_T1126, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1128 = cond(X_T1127, X_T36, X_T1126)
// Elementwise op: [[pid(Sqrt)]] X_T1129 = sqrt(X_T1128)
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
__kernel void kernel_c108_sdk_379(__global float* restrict  X_T1129, __global const float* restrict  X_I_433)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 768) || (i1_tid < 192));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_433 = X_I_433[gout_idx];
    float LX_T1126 = (1.0009999641624745e-5f + LX_I_433);
    int LX_T1127 = (LX_T1126 < (float)0);
    float LX_T1128 = select((float)LX_T1126, (float)0, (int)LX_T1127);
    float LX_T1129 = native_sqrt(LX_T1128);
    X_T1129[gout_idx] = LX_T1129;
  }
}
