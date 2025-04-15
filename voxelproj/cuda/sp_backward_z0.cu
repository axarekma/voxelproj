#include "projection_common.cuh"

#define d_MAX(x, y) (((x) > (y)) ? (x) : (y))
#define d_MIN(x, y) (((x) < (y)) ? (x) : (y))

#define MAX_ANGLES 256

extern "C"
{
    __global__ void sp_backward_z0(cudaTextureObject_t tex_proj, float *volume, float *angles,
                                   int p0, int p1, int p2, int v0, int v1, int v2)
    {
        int zi = blockIdx.x * blockDim.x + threadIdx.x;
        int xi = blockIdx.y * blockDim.y + threadIdx.y;
        int yi = blockIdx.z * blockDim.z + threadIdx.z;

        // Memory layout
        // VOLUME in (height,slice_x,slice_y)
        // PROJECTIONS in (height,px,angle)
        int const nz = v0;
        int const nx = v1;
        int const ny = v2;
        int const px = p1;
        int const pa = p2;

        // if (zi == 0 && xi == 0 && yi == 0)
        // {
        //     printf(" === sp_backward_z0 ===\n");
        //     printf("Nproj: (Height,Width, angles) (%d, %d, %d) \n", Nproj.x, Nproj.y, Nproj.z, Nproj.w);
        //     printf("Nvol: (Height,Sx,Sy) (%d, %d, %d) \n", Nvol.x, Nvol.y, Nvol.z, Nvol.w);
        //     printf("Angles: first = %f, last = %f \n", angles[0], angles[Nproj.z - 1]);
        // }

        float const xcent = 0.5f * (nx - 1);
        float const ycent = 0.5f * (ny - 1);
        float const pcent = 0.5f * (px - 1);

        // Define shared memory for angle calculations
        __shared__ float2 s_cossin[MAX_ANGLES];
        __shared__ float3 s_pwi_args[MAX_ANGLES];
        // Thread block indices
        int const tx = threadIdx.x;
        int const ty = threadIdx.y;
        int const tz = threadIdx.z;
        // Calculate local thread ID within the block
        int const tid = tz * blockDim.x * blockDim.y + ty * blockDim.x + tx;
        // Calculate total threads per block
        int const totalThreadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
        // Load angles into shared memory cooperatively
        for (int i = tid; i < pa; i += totalThreadsPerBlock)
        {
            if (i < pa)
            {

                s_cossin[i] = make_float2(cos(angles[i]), sin(angles[i]));

                float cms = fabs(0.5f * (s_cossin[i].x - s_cossin[i].y));
                float cps = fabs(0.5f * (s_cossin[i].x + s_cossin[i].y));

                float xmin = d_MIN(cms, cps);
                float xmax = d_MAX(cms, cps);
                float lmax = 1.0f / (((xmax - xmin) + 2.0f * xmin));

                s_pwi_args[i] = make_float3(xmin, xmax, lmax);
            }
        }

        // Make sure all threads have loaded the shared memory
        __syncthreads();

        if (zi >= nz || xi >= nx || yi >= ny)
        {
            return;
        }
        float acc = 0.0f;
        for (int ai = 0; ai < pa; ++ai)
        {

            float2 const cossinphi = s_cossin[ai];
            float3 const pwi_args = s_pwi_args[ai];

            float const cosphi = cossinphi.x;
            float const sinphi = cossinphi.y;

            float const xp = (xi - xcent) * cosphi + (yi - ycent) * sinphi + pcent;
            int const iqx = static_cast<int>(round(xp));
            float const xv = iqx - xp;
            float const val1 = spiece_wise_integrated(xv - 0.5f, pwi_args.x, pwi_args.y, pwi_args.z);
            float const val2 = spiece_wise_integrated(xv + 0.5f, pwi_args.x, pwi_args.y, pwi_args.z);
            float const wx0 = static_cast<float>(val1);
            float const wx1 = static_cast<float>(val2 - val1);
            float const wx2 = static_cast<float>(1.0 - val2);

            if (iqx - 1 >= 0 && iqx - 1 < px)
            {
                acc += wx0 * tex3D<float>(tex_proj, zi, iqx - 1, ai);
            }

            if (iqx >= 0 && iqx < px)
            {
                acc += wx1 * tex3D<float>(tex_proj, zi, iqx, ai);
            }
            if (iqx + 1 >= 0 && iqx + 1 < px)
            {
                acc += wx2 * tex3D<float>(tex_proj, zi, iqx + 1, ai);
            }
        }
        int const vol_index = v0 * (v1 * yi + xi) + zi;
        volume[vol_index] += acc;
        return;
    }
}

#undef d_MAX
#undef d_MIN
#undef MAX_ANGLES
