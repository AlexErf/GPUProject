#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 992 }
// Out stride: { 1 }
// Elementwise input X_I_443 shape: fp32(992):(1):3.875 KiB
// Elementwise op: [[pid(Add)]] X_T1131 = add(X_T43, X_I_443)
// Elementwise op: X_T1132 = cmp_lt(X_T1131, X_T20)
// Elementwise op: [[pid(Sqrt)]] X_T1133 = cond(X_T1132, X_T20, X_T1131)
// Elementwise op: [[pid(Sqrt)]] X_T1134 = sqrt(X_T1133)
// Tile size: { 256 }
// Contraction output var shape: fp32(992):(1):3.875 KiB
// Computed true ops: 3968
// Computed work groups: 4
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1024, 1, 1
__kernel void kernel_c68_sdk_388(__global float* restrict  X_T1134, __global const float* restrict  X_I_443)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 768) || (i1_tid < 224));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_443 = X_I_443[gout_idx];
    float LX_T1131 = (1.0009999641624745e-5f + LX_I_443);
    int LX_T1132 = (LX_T1131 < (float)0);
    float LX_T1133 = select((float)LX_T1131, (float)0, (int)LX_T1132);
    float LX_T1134 = native_sqrt(LX_T1133);
    X_T1134[gout_idx] = LX_T1134;
  }
}
