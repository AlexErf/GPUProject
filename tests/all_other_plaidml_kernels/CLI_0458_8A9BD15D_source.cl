#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1216 }
// Out stride: { 1 }
// Elementwise input X_I_513 shape: fp32(1216):(1):4.75 KiB
// Elementwise op: [[pid(Add)]] X_T1326 = add(X_T63, X_I_513)
// Elementwise op: X_T1327 = cmp_lt(X_T1326, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1328 = cond(X_T1327, X_T36, X_T1326)
// Elementwise op: [[pid(Sqrt)]] X_T1329 = sqrt(X_T1328)
// Tile size: { 256 }
// Contraction output var shape: fp32(1216):(1):4.75 KiB
// Computed true ops: 4864
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
__kernel void kernel_c108_sdk_451(__global float* restrict  X_T1329, __global const float* restrict  X_I_513)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1024) || (i1_tid < 192));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_513 = X_I_513[gout_idx];
    float LX_T1326 = (1.0009999641624745e-5f + LX_I_513);
    int LX_T1327 = (LX_T1326 < (float)0);
    float LX_T1328 = select((float)LX_T1326, (float)0, (int)LX_T1327);
    float LX_T1329 = native_sqrt(LX_T1328);
    X_T1329[gout_idx] = LX_T1329;
  }
}
