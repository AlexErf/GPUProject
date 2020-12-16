#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1664 }
// Out stride: { 1 }
// Elementwise input X_I_653 shape: fp32(1664):(1):6.5 KiB
// Elementwise op: [[pid(Add)]] X_T1684 = add(X_T71, X_I_653)
// Elementwise op: X_T1685 = cmp_lt(X_T1684, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1686 = cond(X_T1685, X_T36, X_T1684)
// Elementwise op: [[pid(Sqrt)]] X_T1687 = sqrt(X_T1686)
// Tile size: { 256 }
// Contraction output var shape: fp32(1664):(1):6.5 KiB
// Computed true ops: 6656
// Computed work groups: 7
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 1, 1
__kernel void kernel_c124_sdk_577(__global float* restrict  X_T1687, __global const float* restrict  X_I_653)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1536) || (i1_tid < 128));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_653 = X_I_653[gout_idx];
    float LX_T1684 = (1.0009999641624745e-5f + LX_I_653);
    int LX_T1685 = (LX_T1684 < (float)0);
    float LX_T1686 = select((float)LX_T1684, (float)0, (int)LX_T1685);
    float LX_T1687 = native_sqrt(LX_T1686);
    X_T1687[gout_idx] = LX_T1687;
  }
}
