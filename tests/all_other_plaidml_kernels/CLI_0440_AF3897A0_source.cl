#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 992 }
// Out stride: { 1 }
// Elementwise input X_I_443 shape: fp32(992):(1):3.875 KiB
// Elementwise op: [[pid(Add)]] X_T1159 = add(X_T71, X_I_443)
// Elementwise op: X_T1160 = cmp_lt(X_T1159, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1161 = cond(X_T1160, X_T36, X_T1159)
// Elementwise op: [[pid(Sqrt)]] X_T1162 = sqrt(X_T1161)
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
__kernel void kernel_c124_sdk_388(__global float* restrict  X_T1162, __global const float* restrict  X_I_443)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 768) || (i1_tid < 224));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_443 = X_I_443[gout_idx];
    float LX_T1159 = (1.0009999641624745e-5f + LX_I_443);
    int LX_T1160 = (LX_T1159 < (float)0);
    float LX_T1161 = select((float)LX_T1159, (float)0, (int)LX_T1160);
    float LX_T1162 = native_sqrt(LX_T1161);
    X_T1162[gout_idx] = LX_T1162;
  }
}
