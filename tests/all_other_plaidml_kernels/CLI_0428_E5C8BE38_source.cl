#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1056 }
// Out stride: { 1 }
// Elementwise input X_I_463 shape: fp32(1056):(1):4.125 KiB
// Elementwise op: [[pid(Add)]] X_T1201 = add(X_T63, X_I_463)
// Elementwise op: X_T1202 = cmp_lt(X_T1201, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1203 = cond(X_T1202, X_T36, X_T1201)
// Elementwise op: [[pid(Sqrt)]] X_T1204 = sqrt(X_T1203)
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
__kernel void kernel_c108_sdk_406(__global float* restrict  X_T1204, __global const float* restrict  X_I_463)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1024) || (i1_tid < 32));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_463 = X_I_463[gout_idx];
    float LX_T1201 = (1.0009999641624745e-5f + LX_I_463);
    int LX_T1202 = (LX_T1201 < (float)0);
    float LX_T1203 = select((float)LX_T1201, (float)0, (int)LX_T1202);
    float LX_T1204 = native_sqrt(LX_T1203);
    X_T1204[gout_idx] = LX_T1204;
  }
}
