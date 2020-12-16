#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1056 }
// Out stride: { 1 }
// Elementwise input X_I_463 shape: fp32(1056):(1):4.125 KiB
// Elementwise op: [[pid(Add)]] X_T1209 = add(X_T71, X_I_463)
// Elementwise op: X_T1210 = cmp_lt(X_T1209, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1211 = cond(X_T1210, X_T36, X_T1209)
// Elementwise op: [[pid(Sqrt)]] X_T1212 = sqrt(X_T1211)
// Tile size: { 256 }
// Contraction output var shape: fp32(1056):(1):4.125 KiB
// Computed true ops: 4224
// Computed work groups: 5
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1280, 1, 1
__kernel void kernel_c124_sdk_406(__global float* restrict  X_T1212, __global const float* restrict  X_I_463)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1024) || (i1_tid < 32));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_463 = X_I_463[gout_idx];
    float LX_T1209 = (1.0009999641624745e-5f + LX_I_463);
    int LX_T1210 = (LX_T1209 < (float)0);
    float LX_T1211 = select((float)LX_T1209, (float)0, (int)LX_T1210);
    float LX_T1212 = native_sqrt(LX_T1211);
    X_T1212[gout_idx] = LX_T1212;
  }
}
