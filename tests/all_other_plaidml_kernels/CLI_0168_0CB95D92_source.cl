#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 8, 8, 320 }
// Out stride: { 20480, 2560, 320, 1 }
// Elementwise input X_T1987 shape: fp32(1, 8, 8, 320):(20480, 2560, 320, 1):80 KiB
// Elementwise input X_T1991 shape: fp32(320):(1):1.25 KiB
// Elementwise input X_I_706 shape: fp32(320):(1):1.25 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1992 = div(X_T1987, X_T1991)
// Elementwise op: [[pid(Add, Switch)]] X_T1993 = add(X_T1992, X_I_706)
// Elementwise op: X_T1994 = cmp_lt(X_T1993, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1995 = cond(X_T1994, X_T2, X_T1993)
// Tile size: { 1, 4, 1, 320 }
// Contraction output var shape: fp32(1, 8, 8, 320):(20480, 2560, 320, 1):80 KiB
// Computed true ops: 81920
// Computed work groups: 16
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 5120
// Computed mem read: 480
// Computed mem write: 5120
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 512, 8, 1
__kernel void kernel_c51_sdk_649(__global float* restrict  X_T1995, __global const float* restrict  X_T1987, __global const float* restrict  X_T1991, __global const float* restrict  X_I_706)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 5; i4_lid += 1)
  {
    int i4 = ((64 * i4_lid) + i4_tid);
    int gout_idx = (((2560 * (i2_gid + i2_tid)) + (320 * i3_gid)) + i4);
    float LX_T1987 = X_T1987[gout_idx];
    float LX_T1991 = X_T1991[i4];
    float LX_I_706 = X_I_706[i4];
    float LX_T1992 = (LX_T1987 / LX_T1991);
    float LX_T1993 = (LX_T1992 + LX_I_706);
    int LX_T1994 = (LX_T1993 < 0.0f);
    float LX_T1995 = select((float)LX_T1993, (float)0.0f, (int)LX_T1994);
    X_T1995[gout_idx] = LX_T1995;
  }
}
