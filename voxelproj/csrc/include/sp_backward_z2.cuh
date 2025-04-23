#include "projection_common.cuh"

#define d_MAX(x, y) (((x) > (y)) ? (x) : (y))
#define d_MIN(x, y) (((x) < (y)) ? (x) : (y))

#define MAX_ANGLES 256

__global__ void sp_backward_z2(cudaTextureObject_t tex_proj, float *volume, float *angles,
                               int p0, int p1, int p2, int v0, int v1, int v2)
{
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    int zi = blockIdx.z * blockDim.z + threadIdx.z;

    // Memory layout
    // VOLUME in (slice_x,slice_y,height)
    // PROJECTIONS in (px,angle,height)
    int const nx = v0;
    int const ny = v1;
    int const nz = v2;
    int const px = p0;
    int const pa = p1;

    // if (zi == 0 && xi == 0 && yi == 0)
    // {
    //     printf("    === sp_backward_z2 ===\n");
    //     printf("    Nproj: (Width, angles,Height) (%d, %d, %d) \n", p0, p1, p2);
    //     printf("    Nvol: (Sx,Sy,Height) (%d, %d, %d) \n", v0, v1, v2);
    //     printf("    Angles: first = %f, last = %f \n", angles[0], angles[p1 - 1]);
    // }

    float const xcent = 0.5f * (nx - 1);
    float const ycent = 0.5f * (ny - 1);
    float const pcent = 0.5f * (px - 1);

    // Define shared memory for angle calculations
    __shared__ float2 s_cossin[MAX_ANGLES];
    // __shared__ float3 s_pwi_args[MAX_ANGLES];
    __shared__ float4 s_pwi_args_coeff[MAX_ANGLES];


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
            float s, c;
            sincosf(angles[i], &s, &c);
            s_cossin[i] = make_float2(c, s);
            float const cms = fabsf(0.5f * (c - s));
            float const cps = fabsf(0.5f * (c + s));
            s_cossin[i] = make_float2(cos(angles[i]), sin(angles[i]));
            // float cms = fabs(0.5f * (s_cossin[i].x - s_cossin[i].y));
            // float cps = fabs(0.5f * (s_cossin[i].x + s_cossin[i].y));

            float xmin = d_MIN(cms, cps);
            float xmax = d_MAX(cms, cps);
            float lmax = 1.0f / (((xmax - xmin) + 2.0f * xmin));

            float const coeff = 0.5f * lmax;
            float const coeffdiv = coeff / (xmax - xmin);

            // s_pwi_args[i] = make_float3(xmin, xmax, lmax);
            s_pwi_args_coeff[i] = make_float4(xmin, xmax, coeff, coeffdiv);
        }
    }

    // Make sure all threads have loaded the shared memory
    __syncthreads();

    if (xi >= nx || yi >= ny || zi >= nz)
    {
        return;
    }
    float acc = 0.0f;
    float const relx = (xi - xcent);
    float const rely = (yi - ycent);
    for (int ai = 0; ai < pa; ++ai)
    {
        float2 const cossinphi = s_cossin[ai];
        // float3 const pwi_args = s_pwi_args[ai];
        float4 const pwi_args_coeff = s_pwi_args_coeff[ai];

        float const cosphi = cossinphi.x;
        float const sinphi = cossinphi.y;

        float const xp = relx * cosphi + rely * sinphi + pcent;

        int const iqx = __float2int_rn(xp);
        float const xv = iqx - xp;
        // float const val1 = spiecewise_integrated(xv - 0.5f, pwi_args.x, pwi_args.y, pwi_args.z);
        // float const val2 = spiece_wise_integrated(xv + 0.5f, pwi_args.x, pwi_args.y, pwi_args.z);
        float const val1 = spiece_wise_integrated_precalc(xv - 0.5f, pwi_args_coeff);
        float const val2 = spiece_wise_integrated_precalc(xv + 0.5f, pwi_args_coeff);
        float const wx0 = val1;
        float const wx1 = val2 - val1;
        float const wx2 = 1.0f - val2;

        // cudaAddressModeBorder is set to cudaAddressModeBorder, this is safe
        acc += wx0 * tex3D<float>(tex_proj, iqx - 1, ai, zi);
        acc += wx1 * tex3D<float>(tex_proj, iqx, ai, zi);
        acc += wx2 * tex3D<float>(tex_proj, iqx + 1, ai, zi);
    }
    int const vol_index = v0 * (v1 * zi + yi) + xi;
    volume[vol_index] += acc;
    return;
}

#undef d_MAX
#undef d_MIN
#undef MAX_ANGLES
