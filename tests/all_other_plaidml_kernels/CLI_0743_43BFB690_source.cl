#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2048 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1824 }
// Out stride: { 1 }
// Elementwise input X_I_984 shape: fp32(1824):(1):7.125 KiB
// Elementwise op: [[pid(Add)]] X_T2529 = add(X_T71, X_I_984)
// Elementwise op: X_T2530 = cmp_lt(X_T2529, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T2531 = cond(X_T2530, X_T36, X_T2529)
// Elementwise op: [[pid(Sqrt)]] X_T2532 = sqrt(X_T2531)
// Tile size: { 256 }
// Contraction output var shape: fp32(1824):(1):7.125 KiB
// Computed true ops: 7296
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
__kernel void kernel_c124_sdk_880(__global float* restrict  X_T2532, __global const float* restrict  X_I_984)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1792) || (i1_tid < 32));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_984 = X_I_984[gout_idx];
    float LX_T2529 = (1.0009999641624745e-5f + LX_I_984);
    int LX_T2530 = (LX_T2529 < (float)0);
    float LX_T2531 = select((float)LX_T2529, (float)0, (int)LX_T2530);
    float LX_T2532 = native_sqrt(LX_T2531);
    X_T2532[gout_idx] = LX_T2532;
  }
}
