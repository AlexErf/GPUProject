#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1472 }
// Out stride: { 1 }
// Elementwise input X_I_593 shape: fp32(1472):(1):5.75 KiB
// Elementwise op: [[pid(Add)]] X_T1534 = add(X_T71, X_I_593)
// Elementwise op: X_T1535 = cmp_lt(X_T1534, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1536 = cond(X_T1535, X_T36, X_T1534)
// Elementwise op: [[pid(Sqrt)]] X_T1537 = sqrt(X_T1536)
// Tile size: { 256 }
// Contraction output var shape: fp32(1472):(1):5.75 KiB
// Computed true ops: 5888
// Computed work groups: 6
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 1, 1
__kernel void kernel_c124_sdk_523(__global float* restrict  X_T1537, __global const float* restrict  X_I_593)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1280) || (i1_tid < 192));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_593 = X_I_593[gout_idx];
    float LX_T1534 = (1.0009999641624745e-5f + LX_I_593);
    int LX_T1535 = (LX_T1534 < (float)0);
    float LX_T1536 = select((float)LX_T1534, (float)0, (int)LX_T1535);
    float LX_T1537 = native_sqrt(LX_T1536);
    X_T1537[gout_idx] = LX_T1537;
  }
}
