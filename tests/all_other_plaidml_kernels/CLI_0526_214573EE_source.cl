#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 960 }
// Out stride: { 47040, 6720, 960, 1 }
// Elementwise input X_T1618 shape: fp32(1, 7, 7, 960):(47040, 6720, 960, 1):183.75 KiB
// Elementwise input X_T1641 shape: fp32(1, 7, 7, 960):(47040, 6720, 960, 1):183.75 KiB
// Elementwise input X_I_633 shape: fp32(960):(1):3.75 KiB
// Elementwise input X_I_632 shape: fp32(960):(1):3.75 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1642 = add(X_T1618, X_T1641)
// Elementwise op: [[pid(Sub)]] X_T1644 = sub(X_T1642, X_I_633)
// Elementwise op: [[pid(Mul)]] X_T1645 = mul(X_T1644, X_I_632)
// Tile size: { 1, 1, 1, 960 }
// Contraction output var shape: fp32(1, 7, 7, 960):(47040, 6720, 960, 1):183.75 KiB
// Computed true ops: 141120
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 480
// Computed mem write: 7680
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c108_sdk_563(__global float* restrict  X_T1642, __global float* restrict  X_T1645, __global const float* restrict  X_T1618, __global const float* restrict  X_T1641, __global const float* restrict  X_I_633, __global const float* restrict  X_I_632)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_tid < 192));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((6720 * i2_gid) + (960 * i3_gid)) + i4);
      float LX_T1618 = X_T1618[gout_idx];
      float LX_T1641 = X_T1641[gout_idx];
      float LX_I_633 = X_I_633[i4];
      float LX_I_632 = X_I_632[i4];
      float LX_T1642 = (LX_T1618 + LX_T1641);
      float LX_T1644 = (LX_T1642 - LX_I_633);
      float LX_T1645 = (LX_T1644 * LX_I_632);
      X_T1642[gout_idx] = LX_T1642;
      X_T1645[gout_idx] = LX_T1645;
    }
  }
}
