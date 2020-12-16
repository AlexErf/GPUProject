#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 864 }
// Out stride: { 42336, 6048, 864, 1 }
// Elementwise input X_T1570 shape: fp32(1, 7, 7, 864):(42336, 6048, 864, 1):165.375 KiB
// Elementwise input X_T1574 shape: fp32(864):(1):3.375 KiB
// Elementwise input X_I_601 shape: fp32(864):(1):3.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1575 = div(X_T1570, X_T1574)
// Elementwise op: [[pid(Add, Switch)]] X_T1576 = add(X_T1575, X_I_601)
// Elementwise op: X_T1577 = cmp_lt(X_T1576, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1578 = cond(X_T1577, X_T2, X_T1576)
// Tile size: { 1, 1, 1, 864 }
// Contraction output var shape: fp32(1, 7, 7, 864):(42336, 6048, 864, 1):165.375 KiB
// Computed true ops: 169344
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 324
// Computed mem write: 3456
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c108_sdk_539(__global float* restrict  X_T1578, __global const float* restrict  X_T1570, __global const float* restrict  X_T1574, __global const float* restrict  X_I_601)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_tid < 96));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((6048 * i2_gid) + (864 * i3_gid)) + i4);
      float LX_T1570 = X_T1570[gout_idx];
      float LX_T1574 = X_T1574[i4];
      float LX_I_601 = X_I_601[i4];
      float LX_T1575 = (LX_T1570 / LX_T1574);
      float LX_T1576 = (LX_T1575 + LX_I_601);
      int LX_T1577 = (LX_T1576 < 0.0f);
      float LX_T1578 = select((float)LX_T1576, (float)0.0f, (int)LX_T1577);
      X_T1578[gout_idx] = LX_T1578;
    }
  }
}
