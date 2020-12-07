kernel void memset(   global uint *dst )
{
<<<<<<< HEAD
        dst[get_global_id(0)] = get_global_id(0);
=======
    dst[get_global_id(0)] = get_global_id(0);
>>>>>>> main
}
