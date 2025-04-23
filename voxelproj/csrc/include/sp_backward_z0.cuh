#include "projection_common.cuh"

#define d_MAX(x, y) (((x) > (y)) ? (x) : (y))
#define d_MIN(x, y) (((x) < (y)) ? (x) : (y))

#define MAX_ANGLES 512

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

    float const xcent = 0.5f * (nx - 1);
    float const ycent = 0.5f * (ny - 1);
    float const pcent = 0.5f * (px - 1);

    // Define shared memory for angle calculations
    __shared__ float2 s_cossin[MAX_ANGLES];
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

            float const xmin = d_MIN(cms, cps);
            float const xmax = d_MAX(cms, cps);
            float const lmax = 1.0f / (((xmax - xmin) + 2.0f * xmin));

            float const coeff = 0.5f * lmax;
            float const coeffdiv = coeff / (xmax - xmin);

            s_pwi_args_coeff[i] = make_float4(xmin, xmax, coeff, coeffdiv);
        }
    }

    // Make sure all threads have loaded the shared memory
    __syncthreads();

    if (zi >= nz || xi >= nx || yi >= ny)
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
        float const val1 = spiece_wise_integrated_precalc(xv - 0.5f, pwi_args_coeff);
        float const val2 = spiece_wise_integrated_precalc(xv + 0.5f, pwi_args_coeff);
        float const wx0 = val1;
        float const wx1 = val2 - val1;
        float const wx2 = 1.0f - val2;

        acc += wx0 * tex3D<float>(tex_proj, zi, iqx - 1, ai);
        acc += wx1 * tex3D<float>(tex_proj, zi, iqx, ai);
        acc += wx2 * tex3D<float>(tex_proj, zi, iqx + 1, ai);
    }
    int const vol_index = v0 * (v1 * yi + xi) + zi;
    volume[vol_index] += acc;
    return;
}

#undef d_MAX
#undef d_MIN
#undef MAX_ANGLES
