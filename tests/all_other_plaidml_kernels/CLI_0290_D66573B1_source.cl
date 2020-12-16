#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 640 }
// Out stride: { 1 }
// Elementwise input X_I_333 shape: fp32(640):(1):2.5 KiB
// Elementwise op: [[pid(Add)]] X_T856 = add(X_T43, X_I_333)
// Elementwise op: X_T857 = cmp_lt(X_T856, X_T20)
// Elementwise op: [[pid(Sqrt)]] X_T858 = cond(X_T857, X_T20, X_T856)
// Elementwise op: [[pid(Sqrt)]] X_T859 = sqrt(X_T858)
// Tile size: { 256 }
// Contraction output var shape: fp32(640):(1):2.5 KiB
// Computed true ops: 2560
// Computed work groups: 3
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 1, 1
__kernel void kernel_c68_sdk_289(__global float* restrict  X_T859, __global const float* restrict  X_I_333)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 512) || (i1_tid < 128));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_333 = X_I_333[gout_idx];
    float LX_T856 = (1.0009999641624745e-5f + LX_I_333);
    int LX_T857 = (LX_T856 < (float)0);
    float LX_T858 = select((float)LX_T856, (float)0, (int)LX_T857);
    float LX_T859 = native_sqrt(LX_T858);
    X_T859[gout_idx] = LX_T859;
  }
}
