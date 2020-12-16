#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1728 }
// Out stride: { 1 }
// Elementwise input X_I_673 shape: fp32(1728):(1):6.75 KiB
// Elementwise op: [[pid(Add)]] X_T1734 = add(X_T71, X_I_673)
// Elementwise op: X_T1735 = cmp_lt(X_T1734, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1736 = cond(X_T1735, X_T36, X_T1734)
// Elementwise op: [[pid(Sqrt)]] X_T1737 = sqrt(X_T1736)
// Tile size: { 256 }
// Contraction output var shape: fp32(1728):(1):6.75 KiB
// Computed true ops: 6912
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
__kernel void kernel_c124_sdk_595(__global float* restrict  X_T1737, __global const float* restrict  X_I_673)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1536) || (i1_tid < 192));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_673 = X_I_673[gout_idx];
    float LX_T1734 = (1.0009999641624745e-5f + LX_I_673);
    int LX_T1735 = (LX_T1734 < (float)0);
    float LX_T1736 = select((float)LX_T1734, (float)0, (int)LX_T1735);
    float LX_T1737 = native_sqrt(LX_T1736);
    X_T1737[gout_idx] = LX_T1737;
  }
}
