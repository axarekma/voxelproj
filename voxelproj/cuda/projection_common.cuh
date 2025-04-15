#ifndef PROJECTION_COMMON_CUH
#define PROJECTION_COMMON_CUH

#include "cuda.h"
#include "cuda_runtime.h"

inline int num_blocks(int length, int divider) { return (length + divider - 1) / divider; }

__device__ __forceinline__ void swap_float(float &a, float &b)
{
    float temp = a;
    a = b;
    b = temp;
}

__device__ __forceinline__ int round_and_clamp_upper(float const x, int const max_val)
{
    int xi = __float2int_rn(x);
    return min(max_val, xi);
}

__device__ __forceinline__ int round_and_clamp_lower(float const x, int const min_val)
{
    int xi = __float2int_rn(x);
    return max(min_val, xi);
}

struct ProjDimensions
{
    int vx;
    int vy;
    int vz;

    int pa;
    int px;
    int py;

    // pointer conv functions for volume (z,x,y) projection (z,px,ai)
    __device__ __host__ __forceinline__ constexpr int volumeIndex(int ix, int iy, int iz) const
    {
        return vz * (vx * iy + ix) + iz;
    }
    __device__ __host__ __forceinline__ constexpr int projectionIndex(int ia, int ix,
                                                                      int iy) const
    {
        return py * (px * ia + ix) + iy;
    }

    // pointer conv functions for volume (x,y,z) projection (px,ai,z)
    __device__ __host__ __forceinline__ constexpr int volumeIndex_z2(int ix, int iy, int iz) const
    {
        return vx * (vy * iz + iy) + ix;
    }
    __device__ __host__ __forceinline__ constexpr int projectionIndex_z2(int ia, int ix,
                                                                         int iy) const
    {
        return px * (pa * iy + ia) + ix;
    }
};

__device__ inline double dpiece_wise_integrated(double x, const double a, const double b,
                                                const double y_max)
{
    double const coeff = 0.5 * y_max;
    if (x <= -b)
    {
        return 0.0;
    }
    else if (x < -a)
    {
        return coeff * (x + b) * (x + b) / (b - a);
    }
    else if (x < a)
    {
        return coeff * (b + a) + y_max * x;
    }
    else if (x < b)
    {
        return coeff * (b - x) * (b - x) / (a - b) + 1.0;
    }
    else
    {
        return 1.0;
    }
}

__device__ float inline spiece_wise_integrated(float x, const float a, const float b,
                                               const float y_max)
{

    // Fast path for common cases
    if (x <= -b)
        return 0.0f;
    if (x >= b)
        return 1.0f;

    float const coeff = 0.5f * y_max;
    if (x < -a)
    {
        return coeff * (x + b) * (x + b) / (b - a);
    }
    else if (x < a)
    {
        return coeff * (b + a) + y_max * x;
    }
    else if (x < b)
    {
        return coeff * (b - x) * (b - x) / (a - b) + 1.0f;
    }
    return 0.0f;
}

__device__ inline float2 SideIntersectVec(float2 a1, float2 d1, float2 b1, float2 d2)
{
    float2 b1ma1 = make_float2(b1.x - a1.x, b1.y - a1.y);
    float d1d1 = d1.x * d1.x + d1.y * d1.y;
    float d1d2 = d1.x * d2.x + d1.y * d2.y;
    float d2d2 = d2.x * d2.x + d2.y * d2.y;

    float b1ma1dotd1 = b1ma1.x * d1.x + b1ma1.y * d1.y;
    float b1ma1dotd2 = b1ma1.x * d2.x + b1ma1.y * d2.y;
    float D = d1d1 * d2d2 - d1d2 * d1d2;
    float Dt1 = b1ma1dotd1 * d2d2 - d1d2 * b1ma1dotd2;
    float Dt2 = b1ma1dotd1 * d1d2 - d1d1 * b1ma1dotd2;

    // Test if the lines are parallel
    if (fabs(D) > 1e-15f)
    {
        // Lines are not parallel, return intersection point
        return make_float2(Dt1 / D, Dt2 / D);
    }
    else
    {
        // Lines are parallel, no intersection
        return make_float2(-1.0f, -1.0f);
    }
}

__device__ inline bool RectangleIntersectVec(float2 p0, float2 vec, float4 bbox,
                                             float4 &intersect_box)
{
    float2 const b0 = {bbox.x, bbox.y};
    float2 const b1 = {bbox.z, bbox.w};
    float2 const line_x = {b1.x - b0.x, 0.0f};
    float2 const line_y = {0.0f, b1.y - b0.y};

    float2 bl = b0;
    float2 br = {b1.x, b0.y};
    float2 tl = {b0.x, b1.y};
    // float2 tr = b1;

    float2 int_bottom = SideIntersectVec(p0, vec, bl, line_x);
    float2 int_top = SideIntersectVec(p0, vec, tl, line_x);
    float2 int_left = SideIntersectVec(p0, vec, bl, line_y);
    float2 int_right = SideIntersectVec(p0, vec, br, line_y);

    float t1_bottom = int_bottom.x, t2_bottom = int_bottom.y;
    float t1_top = int_top.x, t2_top = int_top.y;
    float t1_left = int_left.x, t2_left = int_left.y;
    float t1_right = int_right.x, t2_right = int_right.y;
    if (0 <= t2_bottom && t2_bottom <= 1)
    {
        intersect_box.x = p0.x + t1_bottom * vec.x;
        intersect_box.y = p0.y + t1_bottom * vec.y;

        if (0 <= t2_top && t2_top <= 1)
        {
            intersect_box.z = p0.x + t1_top * vec.x;
            intersect_box.w = p0.y + t1_top * vec.y;
            return true;
        }
        if (0 <= t2_left && t2_left <= 1)
        {
            intersect_box.z = p0.x + t1_left * vec.x;
            intersect_box.w = p0.y + t1_left * vec.y;
            return true;
        }
        if (0 <= t2_right && t2_right <= 1)
        {
            intersect_box.z = p0.x + t1_right * vec.x;
            intersect_box.w = p0.y + t1_right * vec.y;
            return true;
        }
    }
    if (0 <= t2_top && t2_top <= 1)
    {
        intersect_box.x = p0.x + t1_top * vec.x;
        intersect_box.y = p0.y + t1_top * vec.y;

        if (0 <= t2_left && t2_left <= 1)
        {
            intersect_box.z = p0.x + t1_left * vec.x;
            intersect_box.w = p0.y + t1_left * vec.y;
            return true;
        }
        if (0 <= t2_right && t2_right <= 1)
        {
            intersect_box.z = p0.x + t1_right * vec.x;
            intersect_box.w = p0.y + t1_right * vec.y;
            return true;
        }
    }
    if (0 <= t2_left && t2_left <= 1)
    {
        intersect_box.x = p0.x + t1_left * vec.x;
        intersect_box.y = p0.y + t1_left * vec.y;

        if (0 <= t2_right && t2_right <= 1)
        {
            intersect_box.z = p0.x + t1_right * vec.x;
            intersect_box.w = p0.y + t1_right * vec.y;
            return true;
        }
    }

    return false;
}

#endif
