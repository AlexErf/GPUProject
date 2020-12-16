#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1600 }
// Out stride: { 1 }
// Elementwise input X_I_633 shape: fp32(1600):(1):6.25 KiB
// Elementwise op: [[pid(Add)]] X_T1634 = add(X_T71, X_I_633)
// Elementwise op: X_T1635 = cmp_lt(X_T1634, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1636 = cond(X_T1635, X_T36, X_T1634)
// Elementwise op: [[pid(Sqrt)]] X_T1637 = sqrt(X_T1636)
// Tile size: { 256 }
// Contraction output var shape: fp32(1600):(1):6.25 KiB
// Computed true ops: 6400
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
__kernel void kernel_c124_sdk_559(__global float* restrict  X_T1637, __global const float* restrict  X_I_633)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1536) || (i1_tid < 64));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_633 = X_I_633[gout_idx];
    float LX_T1634 = (1.0009999641624745e-5f + LX_I_633);
    int LX_T1635 = (LX_T1634 < (float)0);
    float LX_T1636 = select((float)LX_T1634, (float)0, (int)LX_T1635);
    float LX_T1637 = native_sqrt(LX_T1636);
    X_T1637[gout_idx] = LX_T1637;
  }
}
