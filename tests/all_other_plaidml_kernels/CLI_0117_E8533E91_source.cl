#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 384 }
// Out stride: { 1 }
// Elementwise input X_I_160 shape: fp32(384):(1):1.5 KiB
// Elementwise op: [[pid(Add)]] X_T333 = add(X_T59, X_I_160)
// Elementwise op: X_T334 = cmp_lt(X_T333, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T335 = cond(X_T334, X_T4, X_T333)
// Elementwise op: [[pid(Sqrt)]] X_T336 = sqrt(X_T335)
// Tile size: { 256 }
// Contraction output var shape: fp32(384):(1):1.5 KiB
// Computed true ops: 1536
// Computed work groups: 2
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 512, 1, 1
__kernel void kernel_c43_sdk_86(__global float* restrict  X_T336, __global const float* restrict  X_I_160)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 256) || (i1_tid < 128));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_160 = X_I_160[gout_idx];
    float LX_T333 = (0.0010000000474974513f + LX_I_160);
    int LX_T334 = (LX_T333 < (float)0);
    float LX_T335 = select((float)LX_T333, (float)0, (int)LX_T334);
    float LX_T336 = native_sqrt(LX_T335);
    X_T336[gout_idx] = LX_T336;
  }
}
