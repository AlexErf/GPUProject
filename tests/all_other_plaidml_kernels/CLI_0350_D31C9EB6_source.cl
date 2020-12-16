#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 640 }
// Out stride: { 1 }
// Elementwise input X_I_333 shape: fp32(640):(1):2.5 KiB
// Elementwise op: [[pid(Add)]] X_T876 = add(X_T63, X_I_333)
// Elementwise op: X_T877 = cmp_lt(X_T876, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T878 = cond(X_T877, X_T36, X_T876)
// Elementwise op: [[pid(Sqrt)]] X_T879 = sqrt(X_T878)
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
__kernel void kernel_c108_sdk_289(__global float* restrict  X_T879, __global const float* restrict  X_I_333)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 512) || (i1_tid < 128));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_333 = X_I_333[gout_idx];
    float LX_T876 = (1.0009999641624745e-5f + LX_I_333);
    int LX_T877 = (LX_T876 < (float)0);
    float LX_T878 = select((float)LX_T876, (float)0, (int)LX_T877);
    float LX_T879 = native_sqrt(LX_T878);
    X_T879[gout_idx] = LX_T879;
  }
}
