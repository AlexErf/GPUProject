#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 448 }
// Out stride: { 1 }
// Elementwise input X_I_192 shape: fp32(448):(1):1.75 KiB
// Elementwise op: [[pid(Add)]] X_T486 = add(X_T43, X_I_192)
// Elementwise op: X_T487 = cmp_lt(X_T486, X_T20)
// Elementwise op: [[pid(Sqrt)]] X_T488 = cond(X_T487, X_T20, X_T486)
// Elementwise op: [[pid(Sqrt)]] X_T489 = sqrt(X_T488)
// Tile size: { 256 }
// Contraction output var shape: fp32(448):(1):1.75 KiB
// Computed true ops: 1792
// Computed work groups: 2
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 512, 1, 1
__kernel void kernel_c68_sdk_157(__global float* restrict  X_T489, __global const float* restrict  X_I_192)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 256) || (i1_tid < 192));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_192 = X_I_192[gout_idx];
    float LX_T486 = (1.0009999641624745e-5f + LX_I_192);
    int LX_T487 = (LX_T486 < (float)0);
    float LX_T488 = select((float)LX_T486, (float)0, (int)LX_T487);
    float LX_T489 = native_sqrt(LX_T488);
    X_T489[gout_idx] = LX_T489;
  }
}
