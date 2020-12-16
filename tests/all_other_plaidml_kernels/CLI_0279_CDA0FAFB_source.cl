#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 384 }
// Out stride: { 1 }
// Elementwise input X_I_172 shape: fp32(384):(1):1.5 KiB
// Elementwise op: [[pid(Add)]] X_T464 = add(X_T71, X_I_172)
// Elementwise op: X_T465 = cmp_lt(X_T464, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T466 = cond(X_T465, X_T36, X_T464)
// Elementwise op: [[pid(Sqrt)]] X_T467 = sqrt(X_T466)
// Tile size: { 256 }
// Contraction output var shape: fp32(384):(1):1.5 KiB
// Computed true ops: 1536
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
__kernel void kernel_c124_sdk_139(__global float* restrict  X_T467, __global const float* restrict  X_I_172)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 256) || (i1_tid < 128));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_172 = X_I_172[gout_idx];
    float LX_T464 = (1.0009999641624745e-5f + LX_I_172);
    int LX_T465 = (LX_T464 < (float)0);
    float LX_T466 = select((float)LX_T464, (float)0, (int)LX_T465);
    float LX_T467 = native_sqrt(LX_T466);
    X_T467[gout_idx] = LX_T467;
  }
}
