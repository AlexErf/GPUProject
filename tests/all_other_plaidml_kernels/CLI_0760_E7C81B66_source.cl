#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2048 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1920 }
// Out stride: { 1 }
// Elementwise input X_I_1006 shape: fp32(1920):(1):7.5 KiB
// Elementwise op: [[pid(Add)]] X_T2603 = add(X_T71, X_I_1006)
// Elementwise op: X_T2604 = cmp_lt(X_T2603, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T2605 = cond(X_T2604, X_T36, X_T2603)
// Elementwise op: [[pid(Sqrt)]] X_T2606 = sqrt(X_T2605)
// Tile size: { 256 }
// Contraction output var shape: fp32(1920):(1):7.5 KiB
// Computed true ops: 7680
// Computed work groups: 8
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2048, 1, 1
__kernel void kernel_c124_sdk_906(__global float* restrict  X_T2606, __global const float* restrict  X_I_1006)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1792) || (i1_tid < 128));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_1006 = X_I_1006[gout_idx];
    float LX_T2603 = (1.0009999641624745e-5f + LX_I_1006);
    int LX_T2604 = (LX_T2603 < (float)0);
    float LX_T2605 = select((float)LX_T2603, (float)0, (int)LX_T2604);
    float LX_T2606 = native_sqrt(LX_T2605);
    X_T2606[gout_idx] = LX_T2606;
  }
}
