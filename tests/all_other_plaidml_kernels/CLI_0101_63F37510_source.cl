#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1 1 1
// lid: 1 1 1
// Original:
// X_T351[i0, 0 : _T515, _T516] = =(X_T350[i0])
// With Index Variables Made Integral:
// X_T351[i0, 0 : _T515, _T516] = =(X_T350[i0]), 500000000 + i0 < 1000000000
// Constraints:{ 0 <= i0 < 1, 0 <= 0 < 1, 0 <= i0 < 1, 0 <= 500000000 + i0 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= i0 < 1 }
// Defracted:
// X_T351[i0, 0 : _T515, _T516] = =(X_T350[i0]), 500000000 + i0 < 1000000000
// Flattened:
//              Range    X_T351    X_T350  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: {  }
// Ranges: {  }
// Out stride: {  }
// Input 1 offset: 0
// Input 1 stride: {  }
// Tile size: {  }
// Contraction output var shape: fp32(1, 1):(1, 1):4 bytes
// Computed true ops: 2
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 1024
// Computed mem read: 128
// Computed mem write: 128
// Computed operations: 1
// Computed rollups: 0
// Computed threads used: 1
// lwork = 1, 1, 1
// gwork = 1, 1, 1
__kernel void kernel_c6_sdk_176(__global float* restrict  X_T351, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[1];
  {
    {
      in1_shared[0] = in1[clamp((char)0, (char)0, (char)0)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float val1 = in1_shared[0];
    agg[0] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float LX_T351 = agg[0];
  X_T351[0] = LX_T351;
}
