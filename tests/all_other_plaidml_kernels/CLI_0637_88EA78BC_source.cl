#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1600 }
// Out stride: { 1 }
// Elementwise input X_I_834 shape: fp32(1600):(1):6.25 KiB
// Elementwise op: [[pid(Add)]] X_T2146 = add(X_T63, X_I_834)
// Elementwise op: X_T2147 = cmp_lt(X_T2146, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T2148 = cond(X_T2147, X_T36, X_T2146)
// Elementwise op: [[pid(Sqrt)]] X_T2149 = sqrt(X_T2148)
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
__kernel void kernel_c108_sdk_745(__global float* restrict  X_T2149, __global const float* restrict  X_I_834)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1536) || (i1_tid < 64));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_834 = X_I_834[gout_idx];
    float LX_T2146 = (1.0009999641624745e-5f + LX_I_834);
    int LX_T2147 = (LX_T2146 < (float)0);
    float LX_T2148 = select((float)LX_T2146, (float)0, (int)LX_T2147);
    float LX_T2149 = native_sqrt(LX_T2148);
    X_T2149[gout_idx] = LX_T2149;
  }
}
