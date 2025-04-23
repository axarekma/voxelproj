#include "projection_common.cuh"

#define d_MAX(x, y) (((x) > (y)) ? (x) : (y))
#define d_MIN(x, y) (((x) < (y)) ? (x) : (y))

namespace order_z0
{
    __device__ float sp_ray_sum_stride0(cudaTextureObject_t tex_vol, const int zi, float2 ray_vector,
                                        float4 bounding_box, const float4 pwi_args, const int4 Nvol)
    {
        int const nx = Nvol.y;
        int const ny = Nvol.z;
        // int const px = Nproj.y;

        float ray_sum = 0.0f;
        float const par_half_width = 0.5f * (1.0f + ray_vector.y) / ray_vector.x;
        // bounding box ( x, y, z, w)
        //              (x0,y0,x1,y1)
        if (bounding_box.x > bounding_box.z)
        {
            // Ensure x direction is positive
            swap_float(bounding_box.x, bounding_box.z);
            swap_float(bounding_box.y, bounding_box.w);
        }
        ray_vector.y = copysignf(ray_vector.y, bounding_box.w - bounding_box.y);

        // dx step_size
        float const dy = ray_vector.y / ray_vector.x;
        // bb detection can 'hit' the extended border before 0
        // The x range is clamped the the nearest voxels inside the image :w
        int const x_start = round_and_clamp_lower(bounding_box.x, 0);
        int const x_stop = round_and_clamp_upper(bounding_box.z, nx - 1);
        float y = bounding_box.y + dy * (x_start - bounding_box.x);

        for (int iqx = x_start; iqx <= x_stop; ++iqx)
        {
            int const y_left = round_and_clamp_lower(y - par_half_width, 0);
            int const y_right = round_and_clamp_upper(y + par_half_width, ny - 1);
            for (int iqy = y_left; iqy <= y_right; ++iqy)
            {
                // float const yv = iqy - y;
                float const yv_proj = (iqy - y) * ray_vector.x;
                float const ray_lb = -0.5f - yv_proj;
                float const ray_rb = 0.5f - yv_proj;
                // float const pwiL = spiece_wise_integrated(ray_lb, pwi_args.x, pwi_args.y, pwi_args.z);
                // float const pwiR = spiece_wise_integrated(ray_rb, pwi_args.x, pwi_args.y, pwi_args.z);
                float const pwiL = spiece_wise_integrated_precalc(ray_lb, pwi_args);
                float const pwiR = spiece_wise_integrated_precalc(ray_rb, pwi_args);

                float const wx = pwiR - pwiL;
                ray_sum += tex3D<float>(tex_vol, zi, iqx, iqy) * wx;
            }
            y += dy; // x takes integer steps, increment y accordingly
        }
        return ray_sum;
    }

    __device__ float sp_ray_sum_stride1(cudaTextureObject_t tex_vol, const int zi, float2 ray_vector,
                                        float4 bounding_box, const float4 pwi_args, const int4 Nvol)
    {
        int const nx = Nvol.y;
        int const ny = Nvol.z;
        // int const px = Nproj.y;

        float ray_sum = 0.0f;
        float const par_half_width = 0.5f * (1.0f + ray_vector.x) / ray_vector.y;
        // bounding box ( x, y, z, w)
        //              (x0,y0,x1,y1)
        if (bounding_box.y > bounding_box.w)
        {
            // Ensure y direction is positive
            swap_float(bounding_box.x, bounding_box.z);
            swap_float(bounding_box.y, bounding_box.w);
        }
        ray_vector.x = copysignf(ray_vector.x, bounding_box.z - bounding_box.x);

        // dx step_size
        float const dx = ray_vector.x / ray_vector.y;
        // bb detection can 'hit' the extended border before 0
        // start stride from y==0
        int const y_start = round_and_clamp_lower(bounding_box.y, 0);
        int const y_stop = round_and_clamp_upper(bounding_box.w, ny - 1);
        float x = bounding_box.x + dx * (y_start - bounding_box.y);

        for (int iqy = y_start; iqy <= y_stop; ++iqy)
        {
            int const x_left = round_and_clamp_lower(x - par_half_width, 0);
            int const x_right = round_and_clamp_upper(x + par_half_width, nx - 1);
            for (int iqx = x_left; iqx <= x_right; ++iqx)
            {
                // float const xv = iqx - x;
                float const xv_proj = (iqx - x) * ray_vector.y;
                float const ray_lb = -0.5f - xv_proj;
                float const ray_rb = 0.5f - xv_proj;
                float const pwiL = spiece_wise_integrated_precalc(ray_lb, pwi_args);
                float const pwiR = spiece_wise_integrated_precalc(ray_rb, pwi_args);
                float const wx = pwiR - pwiL;
                ray_sum += tex3D<float>(tex_vol, zi, iqx, iqy) * wx;
            }
            x += dx; // y takes integer steps, increment x accordingly
        }
        return ray_sum;
    }

}

__global__ void sp_forward_z0(cudaTextureObject_t tex_vol, float *output, float *angles,
                              int v0, int v1, int v2, int p0, int p1, int p2)
{

    int zi = blockIdx.x * blockDim.x + threadIdx.x;
    int pi = blockIdx.y * blockDim.y + threadIdx.y;
    int ai = blockIdx.z * blockDim.z + threadIdx.z;
    // if (zi == 0 && pi == 0 && ai == 0)
    // {
    //     printf(" === sp_forward_z0 ===\n");
    //     printf("Nproj: (Height,Width, Angles) (%d, %d, %d) \n", p0, p1, p2);
    //     printf("Nvol: (Height,Sx,Sy) (%d, %d, %d) \n", v0, v1, v2);
    //     printf("Angles: first = %f, last = %f \n", angles[0], angles[p2 - 1]);
    // }
    int4 Nvol = make_int4(v0, v1, v2, 0);
    if (zi >= p0 || pi >= p1 || ai >= p2)
    {
        // out of bounds check
        return;
    }
    int const index_proj = p0 * (p1 * ai + pi) + zi;
    int const nx = v1;
    int const ny = v2;
    int const px = p1;

    float const xcent = 0.5f * (nx - 1);
    float const ycent = 0.5f * (ny - 1);
    float const pcent = 0.5f * (px - 1);

    float const cosphi = cos(angles[ai]);
    float const sinphi = sin(angles[ai]);
    float const cms = fabs(0.5f * (cosphi - sinphi));
    float const cps = fabs(0.5f * (cosphi + sinphi));

    float const xmin = min(cms, cps);
    float const xmax = max(cms, cps);
    float const lmax = 1 / (((xmax - xmin) + 2 * xmin));
    
    float const coeff = 0.5f * lmax;
    float const coeffdiv = coeff / (xmax - xmin);

    // float3 const pwi_args = make_float3(xmin, xmax, lmax);
    float4 const pwi_args = make_float4(xmin, xmax, coeff, coeffdiv);

    float const vecx = fabs(sinphi);
    float const vecy = fabs(cosphi);
    float2 const v_norm = make_float2(-sinphi, cosphi);
    float2 const ray0 = make_float2((pi - pcent) * cosphi + xcent, (pi - pcent) * sinphi + ycent);
    float2 const ray_vec = make_float2(vecx, vecy);
    float const ray_width = 0.5f + 0.5f / d_MAX(vecx, vecy);

    if (vecy > vecx)
    {
        // y step is bigger, stride along y (index 1)
        float4 const extended_box =
            make_float4(-ray_width, -0.5f, nx - 1 + ray_width, ny - 0.5f);

        float4 intersect_box;
        if (RectangleIntersectVec(ray0, v_norm, extended_box, intersect_box))
        {
            output[index_proj] += order_z0::sp_ray_sum_stride1(tex_vol, zi, ray_vec, intersect_box, pwi_args, Nvol);
            // return 0.0f;
        }
    }
    else
    {
        // x step is bigger, stride along x (index =0 )
        float4 const extended_box =
            make_float4(-0.5, -ray_width, nx - 0.5, ny - 1 + ray_width);

        float4 intersect_box;
        if (RectangleIntersectVec(ray0, v_norm, extended_box, intersect_box))
        {
            output[index_proj] += order_z0::sp_ray_sum_stride0(tex_vol, zi, ray_vec, intersect_box, pwi_args, Nvol);
            // return 0.0f;
        }
    }
    return;
}

#undef d_MAX
#undef d_MIN