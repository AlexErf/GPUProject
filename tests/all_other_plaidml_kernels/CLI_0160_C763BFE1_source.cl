#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 288 }
// Out stride: { 1 }
// Elementwise input X_I_705 shape: fp32(288):(1):1.125 KiB
// Elementwise op: [[pid(Add)]] X_T1952 = add(X_T33, X_I_705)
// Elementwise op: X_T1953 = cmp_lt(X_T1952, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T1954 = cond(X_T1953, X_T4, X_T1952)
// Elementwise op: [[pid(Sqrt)]] X_T1955 = sqrt(X_T1954)
// Tile size: { 128 }
// Contraction output var shape: fp32(288):(1):1.125 KiB
// Computed true ops: 1152
// Computed work groups: 3
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 16
// Computed mem write: 512
// Computed operations: 128
// Computed rollups: 0
// Computed threads used: 128
// lwork = 256, 1, 1
// gwork = 768, 1, 1
__kernel void kernel_c51_sdk_637(__global float* restrict  X_T1955, __global const float* restrict  X_I_705)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 128);
  int i1_tid = (tid % 128);
  int i1_cond = ((i1_gid != 256) || (i1_tid < 32));
  if (i1_cond)
  {
    if ((tid < 128))
    {
      int gout_idx = (i1_gid + i1_tid);
      float LX_I_705 = X_I_705[gout_idx];
      float LX_T1952 = (0.0010000000474974513f + LX_I_705);
      int LX_T1953 = (LX_T1952 < (float)0);
      float LX_T1954 = select((float)LX_T1952, (float)0, (int)LX_T1953);
      float LX_T1955 = native_sqrt(LX_T1954);
      X_T1955[gout_idx] = LX_T1955;
    }
  }
}
