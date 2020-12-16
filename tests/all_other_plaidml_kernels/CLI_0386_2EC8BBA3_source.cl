#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 704 }
// Out stride: { 1 }
// Elementwise input X_I_353 shape: fp32(704):(1):2.75 KiB
// Elementwise op: [[pid(Add)]] X_T934 = add(X_T71, X_I_353)
// Elementwise op: X_T935 = cmp_lt(X_T934, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T936 = cond(X_T935, X_T36, X_T934)
// Elementwise op: [[pid(Sqrt)]] X_T937 = sqrt(X_T936)
// Tile size: { 256 }
// Contraction output var shape: fp32(704):(1):2.75 KiB
// Computed true ops: 2816
// Computed work groups: 3
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 1, 1
__kernel void kernel_c124_sdk_307(__global float* restrict  X_T937, __global const float* restrict  X_I_353)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 512) || (i1_tid < 192));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_353 = X_I_353[gout_idx];
    float LX_T934 = (1.0009999641624745e-5f + LX_I_353);
    int LX_T935 = (LX_T934 < (float)0);
    float LX_T936 = select((float)LX_T934, (float)0, (int)LX_T935);
    float LX_T937 = native_sqrt(LX_T936);
    X_T937[gout_idx] = LX_T937;
  }
}
