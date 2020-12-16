#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 144 }
// Out stride: { 1 }
// Elementwise input X_I_109 shape: fp32(144):(1):576 bytes
// Elementwise op: [[pid(Add)]] X_T139 = add(X_T59, X_I_109)
// Elementwise op: X_T140 = cmp_lt(X_T139, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T141 = cond(X_T140, X_T4, X_T139)
// Elementwise op: [[pid(Sqrt)]] X_T142 = sqrt(X_T141)
// Tile size: { 144 }
// Contraction output var shape: fp32(144):(1):576 bytes
// Computed true ops: 576
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 20
// Computed mem write: 640
// Computed operations: 144
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 256, 1, 1
__kernel void kernel_c43_sdk_31(__global float* restrict  X_T142, __global const float* restrict  X_I_109)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  int i1_cond = (i1_tid < 144);
  if (i1_cond)
  {
    float LX_I_109 = X_I_109[i1_tid];
    float LX_T139 = (0.0010000000474974513f + LX_I_109);
    int LX_T140 = (LX_T139 < (float)0);
    float LX_T141 = select((float)LX_T139, (float)0, (int)LX_T140);
    float LX_T142 = native_sqrt(LX_T141);
    X_T142[i1_tid] = LX_T142;
  }
}
