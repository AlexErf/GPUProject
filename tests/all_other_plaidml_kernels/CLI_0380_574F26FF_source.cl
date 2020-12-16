#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 800 }
// Out stride: { 1 }
// Elementwise input X_I_383 shape: fp32(800):(1):3.125 KiB
// Elementwise op: [[pid(Add)]] X_T1001 = add(X_T63, X_I_383)
// Elementwise op: X_T1002 = cmp_lt(X_T1001, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1003 = cond(X_T1002, X_T36, X_T1001)
// Elementwise op: [[pid(Sqrt)]] X_T1004 = sqrt(X_T1003)
// Tile size: { 256 }
// Contraction output var shape: fp32(800):(1):3.125 KiB
// Computed true ops: 3200
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
__kernel void kernel_c108_sdk_334(__global float* restrict  X_T1004, __global const float* restrict  X_I_383)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 768) || (i1_tid < 32));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_383 = X_I_383[gout_idx];
    float LX_T1001 = (1.0009999641624745e-5f + LX_I_383);
    int LX_T1002 = (LX_T1001 < (float)0);
    float LX_T1003 = select((float)LX_T1001, (float)0, (int)LX_T1002);
    float LX_T1004 = native_sqrt(LX_T1003);
    X_T1004[gout_idx] = LX_T1004;
  }
}
