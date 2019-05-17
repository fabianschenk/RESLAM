#pragma once
#include "../config/Defines.h"
#include "../GUI/MinimalImage.h"
#include <opencv2/core/mat.hpp>

namespace RESLAM
{
//bounds: [minX,minY,maxX,maxY]
//for isInImage(width,height,ptXY) this would look something like this:
// isInImage(Eigen::Vector4f(0,0,width-1,height-1),ptXY)
EIGEN_ALWAYS_INLINE bool 
isInImage(const float width, const float height, const float x, const float y)
{
    return (x >= 0 && y >= 0 && x <= width - 1 && y <= height -1);
}

EIGEN_ALWAYS_INLINE auto 
isInImage(const Vec4f& bounds, const Vec2f& ptXY)
{
    return (ptXY[0] >= bounds[0] && ptXY[1] >= bounds[1] && ptXY[0] <= bounds[2] && ptXY[1] <= bounds[3]);
}

EIGEN_ALWAYS_INLINE auto 
isInImageGreater(const Vec4f& bounds, const Vec2f& ptXY)
{
    return (ptXY[0] > bounds[0] && ptXY[1] > bounds[1] && ptXY[0] < bounds[2] && ptXY[1] < bounds[3]);
}

EIGEN_ALWAYS_INLINE auto 
projectPixel(const float& hostX, const float& hostY, const CameraMatrix& camMat, const float& idepth,
             const Mat33f& R, const Vec3f t, float& new_idepth, 
             Vec2f& pt2DnoK, Vec2f& pt2D, Vec3f& KliP, float& drescale)

{
    KliP = Vec3f((hostX-camMat.cxl())*camMat.fxli(),(hostY-camMat.cyl())*camMat.fyli(),1); //Klip
    Eigen::Vector3f ptp = R*KliP+t*idepth; //ptp
    drescale = 1.0f/ptp[2];
    new_idepth = idepth*drescale;
    if (!(drescale > 0)) return false;
    pt2DnoK = Vec2f(ptp[0]*drescale,ptp[1]*drescale); //u,v
    pt2D = Vec2f(pt2DnoK[0]*camMat.fxl()+camMat.cxl(),pt2DnoK[1]*camMat.fyl()+camMat.cyl()); //Ku,Kv
    // return (isInImageGreater(Eigen::Vector4f(1.0f,1.0f,camMat.wBounds[0],camMat.hBounds[0]),pt2D) && new_idepth > 0);
    return (isInImageGreater(camMat.imageBounds[0],pt2D) && new_idepth > 0);
}

// Note that most of the getInterpolatedXX functions are from LSD-SLAM/DSO

/**
 * Used for interpolating not just the DT but also the gradients
 */
EIGEN_ALWAYS_INLINE Vec3f
getInterpolatedElement33(const Vec3f* const mat, const float x, const float y, const int width)
{
    const int ix = static_cast<int>(x);
    const int iy = static_cast<int>(y);
    const float dx = x - ix;
    const float dy = y - iy;
    const float dxdy = dx*dy;
    const Vec3f* bp = mat +ix+iy*width;
    return dxdy * *(const Vec3f*)(bp+1+width)
            + (dy-dxdy) * *(const Vec3f*)(bp+width)
            + (dx-dxdy) * *(const Vec3f*)(bp+1)
            + (1-dx-dy+dxdy) * *(const Vec3f*)(bp);
}
/**
 * Used for interpolating not just the DT but also the gradients
 */
EIGEN_ALWAYS_INLINE Vec3f 
getInterpolatedElement33(const Vec3f* const mat, const Vec2f& pt, const int width)
{
    return getInterpolatedElement33(mat,pt[0],pt[1],width);
}
/**
 * Used for interpolating the DT on an NxMx3 image
 */
EIGEN_ALWAYS_INLINE float 
getInterpolatedElement31(const Vec3f* const mat, const float x, const float y, const int width)
{
    const int ix = static_cast<int>(x);
    const int iy = static_cast<int>(y);
    const float dx = x - ix;
    const float dy = y - iy;
    const float dxdy = dx*dy;
    const Vec3f* bp = mat +ix+iy*width;
    return dxdy * (*(const Vec3f*)(bp+1+width))[0]
            + (dy-dxdy) * (*(const Vec3f*)(bp+width))[0]
            + (dx-dxdy) * (*(const Vec3f*)(bp+1))[0]
            + (1-dx-dy+dxdy) * (*(const Vec3f*)(bp))[0];
}

EIGEN_ALWAYS_INLINE auto 
getWeightOfEvoR(const float r, const float delta_huber)
{
    //residual is always positive, thus no std::abs needed!
    return (r <= delta_huber ? 1 : delta_huber/r);
}


EIGEN_ALWAYS_INLINE auto 
isEdge(uint8_t val)
{
    //1 if "binary" edge and 255 if "white"
    return (val == 1 || val == 255);
}

EIGEN_ALWAYS_INLINE auto 
isValidDepth(float depthVal, float minDepth = 0.1f, float maxDepth = 5.2f)
{
    // I3D_LOG(i3d::info) << "depthVal: " << depthVal << " min: " << minDepth << " max: " << maxDepth << "valid: " << (depthVal > minDepth && depthVal < maxDepth);
    return (depthVal > minDepth && depthVal < maxDepth);
}

EIGEN_ALWAYS_INLINE auto 
computeEdgeWeight(const cv::Mat& gX, const cv::Mat& gY, int x, int y)
{
    const auto gXVal = gX.at<unsigned short>(y,x);
    const auto gYVal = gY.at<unsigned short>(y,x);

    const auto edgeWeight = std::sqrt(gXVal*gXVal+gYVal*gYVal);
    return edgeWeight;
}

/**
 * Absolute difference of two size_t values
 */
EIGEN_ALWAYS_INLINE auto 
absDiffSizeT(const size_t v1, const size_t v2)
{
    return (v1 < v2 ? v2 - v1 : v1 - v2);
}


inline Vec3b 
makeJet3B(float id)
{
    if(id <= 0) return Vec3b(128,0,0);
    if(id >= 1) return Vec3b(0,0,128);

    int icP = (id*8);
    float ifP = (id*8)-icP;

    if(icP == 0) return Vec3b(255*(0.5+0.5*ifP), 		    		  0,     					0);
    if(icP == 1) return Vec3b(255, 					  255*(0.5*ifP),     					0);
    if(icP == 2) return Vec3b(255, 				  255*(0.5+0.5*ifP),     					0);
    if(icP == 3) return Vec3b(255*(1-0.5*ifP), 					255,     					255*(0.5*ifP));
    if(icP == 4) return Vec3b(255*(0.5-0.5*ifP), 					255,     					255*(0.5+0.5*ifP));
    if(icP == 5) return Vec3b(0, 						255*(1-0.5*ifP),     					255);
    if(icP == 6) return Vec3b(0, 						255*(0.5-0.5*ifP),     					255);
    if(icP == 7) return Vec3b(0, 					  				  0,     					255*(1-0.5*ifP));
    return Vec3b(255,255,255);
}

}
