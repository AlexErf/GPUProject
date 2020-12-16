#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1888 }
// Out stride: { 92512, 13216, 1888, 1 }
// Elementwise input X_T2578 shape: fp32(1, 7, 7, 1888):(92512, 13216, 1888, 1):361.375 KiB
// Elementwise input X_T2582 shape: fp32(1888):(1):7.375 KiB
// Elementwise input X_I_1001 shape: fp32(1888):(1):7.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2583 = div(X_T2578, X_T2582)
// Elementwise op: [[pid(Add, Switch)]] X_T2584 = add(X_T2583, X_I_1001)
// Elementwise op: X_T2585 = cmp_lt(X_T2584, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2586 = cond(X_T2585, X_T2, X_T2584)
// Tile size: { 1, 1, 1, 1888 }
// Contraction output var shape: fp32(1, 7, 7, 1888):(92512, 13216, 1888, 1):361.375 KiB
// Computed true ops: 370048
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 8192
// Computed mem read: 708
// Computed mem write: 7552
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_899(__global float* restrict  X_T2586, __global const float* restrict  X_T2578, __global const float* restrict  X_T2582, __global const float* restrict  X_I_1001)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 8; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 7) || (i4_tid < 96));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((13216 * i2_gid) + (1888 * i3_gid)) + i4);
      float LX_T2578 = X_T2578[gout_idx];
      float LX_T2582 = X_T2582[i4];
      float LX_I_1001 = X_I_1001[i4];
      float LX_T2583 = (LX_T2578 / LX_T2582);
      float LX_T2584 = (LX_T2583 + LX_I_1001);
      int LX_T2585 = (LX_T2584 < 0.0f);
      float LX_T2586 = select((float)LX_T2584, (float)0.0f, (int)LX_T2585);
      X_T2586[gout_idx] = LX_T2586;
    }
  }
}
