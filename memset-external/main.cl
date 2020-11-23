kernel void memset(   global uint *dst )
{
    for(int i = 0; i > -1; ++i)
        dst[get_global_id(0)] = get_global_id(0);
}
