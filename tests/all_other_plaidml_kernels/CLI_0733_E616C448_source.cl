#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1760 }
// Out stride: { 86240, 12320, 1760, 1 }
// Elementwise input X_T2478 shape: fp32(1, 7, 7, 1760):(86240, 12320, 1760, 1):336.875 KiB
// Elementwise input X_T2482 shape: fp32(1760):(1):6.875 KiB
// Elementwise input X_I_961 shape: fp32(1760):(1):6.875 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2483 = div(X_T2478, X_T2482)
// Elementwise op: [[pid(Add, Switch)]] X_T2484 = add(X_T2483, X_I_961)
// Elementwise op: X_T2485 = cmp_lt(X_T2484, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2486 = cond(X_T2485, X_T2, X_T2484)
// Tile size: { 1, 1, 1, 1760 }
// Contraction output var shape: fp32(1, 7, 7, 1760):(86240, 12320, 1760, 1):336.875 KiB
// Computed true ops: 344960
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 660
// Computed mem write: 7040
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_863(__global float* restrict  X_T2486, __global const float* restrict  X_T2478, __global const float* restrict  X_T2482, __global const float* restrict  X_I_961)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 6) || (i4_tid < 224));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((12320 * i2_gid) + (1760 * i3_gid)) + i4);
      float LX_T2478 = X_T2478[gout_idx];
      float LX_T2482 = X_T2482[i4];
      float LX_I_961 = X_I_961[i4];
      float LX_T2483 = (LX_T2478 / LX_T2482);
      float LX_T2484 = (LX_T2483 + LX_I_961);
      int LX_T2485 = (LX_T2484 < 0.0f);
      float LX_T2486 = select((float)LX_T2484, (float)0.0f, (int)LX_T2485);
      X_T2486[gout_idx] = LX_T2486;
    }
  }
}
