/**
* This file is part of RESLAM.
*
* Copyright (C) 2014-2019 Schenk Fabian <schenk at icg dot tugraz dot at> (Graz University of Technology)
* For more information see <https://github.com/fabianschenk/RESLAM/>
*
* RESLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* RESLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with RESLAM. If not, see <http://www.gnu.org/licenses/>.
* 
*
*  If you use this software please cite at least one of the following publications:
*  - RESLAM: A robust edge-based SLAM system,  Schenk Fabian, Fraundorfer Friedrich, ICRA 2019
*  - Robust Edge-based Visual Odometry using Machine-Learned Edges, Schenk Fabian, Fraundorfer Friedrich, IROS 2017
*  - Combining Edge Images and Depth Maps for Robust Visual Odometry, Schenk Fabian, Fraundorfer Friedrich, BMVC 2017
*/
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "DataStructures.h"
#include "../Utils/Logging.h"
#include "../Utils/utility_functions.h"
#include "../Utils/timer.h"
#include "../Utils/CoarseDistanceMap.h"
#include "../System/LocalMapper.h"
#include "../System/SystemSettings.h"
#include "../System/WindowedOptimizer.h"

namespace RESLAM
{
int EdgeFrameResidual::instanceCounter = 0;

FrameSet::FrameSet(const cv::Mat& gray_, const cv::Mat& depth_, const cv::Mat& rgb_,
                   double _timestamp, const SystemSettings& config): mTimestamp(_timestamp),depthJet(depth_.cols, depth_.rows)
{
    colorImg = rgb_.clone();
    depth = depth_.clone();
    float min{FLOAT_INF}, max{0};
    constexpr float MinValidDepth{0.1};
    for (auto x = 0; x < depth.cols;++x)
        for (auto y = 0; y < depth.rows;++y)
        {
            const auto depthVal = depth.at<float>(y,x);
            //search min and max
            if (depthVal > MinValidDepth && depthVal < min) min = depthVal;
            if (depthVal > MinValidDepth && depthVal > max) max = depthVal;
        }

    const Vec3b ColorBlack(0,0,0);
    for (auto x = 0; x < depth.cols;++x)
        for (auto y = 0; y < depth.rows;++y)
        {
            const auto depthVal = depth.at<float>(y,x);
            if (isValidDepth(depthVal))
            {
                //this is the standard image normalization but a lot of terms cancel out since they are either 1 or 0
                const auto normalizedDepth = (depthVal-min)/(max-min);
                depthJet.setPixel1(x,y,makeJet3B(normalizedDepth));
            }
            else depthJet.setPixel1(x,y,ColorBlack);
        }
    //Edge detection    
    if (config.InputSmoothEdgeImage)
    {
        cv::GaussianBlur(gray_,grayScaleImg,cv::Size(3,3),2);
    }
    else
    {
        grayScaleImg = gray_.clone();
    }
    grayScaleImg.convertTo(grayScaleImgF,CV_32FC1);
    if (config.InputComputeGradientsForEdgeDetector)
    {
        cv::Mat gradX,gradY;
        cv::Sobel(grayScaleImg,gradX,CV_16S,1,0,3,1,0);
        cv::Sobel(grayScaleImg,gradY,CV_16S,0,1,3,1,0);
        cv::Canny(gradX,gradY,edgeImage,config.InputCannyEdgeTh1, config.InputCannyEdgeTh2,true);
    }
    else
    {
        cv::Canny(grayScaleImg,edgeImage,config.InputCannyEdgeTh1, config.InputCannyEdgeTh2,3,true);
    }
}
void 
FrameData::makeKeyFrame(const CameraMatrix& camMat)
{
    
    mFrameHeader->makeKeyFrame();
    mFrameHeader->setRefFrame(this);
    I3D_LOG(i3d::info) << "Making " << std::fixed << mFrameSet->mTimestamp << " to keyframe with id: " << mKeyFrameId;
    //Compute the distance transform if it is not already computed
    computeOptimizationStructure(camMat);
    computeValidEdgePixels();
}

FrameData::~FrameData()
{
    if (mOptStructureAlreadyComputed)
    {
        //free the optimization structures
        for (size_t lvl = 0; lvl < PyramidLevels; ++lvl)
        {
            I3D_LOG(i3d::detail) << "Freeing: " << lvl << " at: " << mOptStructures[lvl];
            Eigen::internal::aligned_free(static_cast<void*>(mOptStructures[lvl]));/*  delete[] optPyr[lvl];*/
            mOptStructures[lvl] = nullptr;
        }
        mOptStructureAlreadyComputed = false;
    }
    if (mFernCode != nullptr)
    {
        delete []mFernCode;
        mFernCode = nullptr;
    }
    release();

}

void 
FrameData::computeOptimizationStructure(const CameraMatrix& camMat)
{
    if (mOptStructureAlreadyComputed) return;
    
    cv::distanceTransform(255-mFrameSet->edgeImage,mFrameSet->distanceTransform,CV_DIST_L2, CV_DIST_MASK_PRECISE);

    // Allocate memory for the optimization structures
    for (size_t lvl = 0; lvl < PyramidLevels; ++lvl)
    {
        //sizeof(Vec3f) = 3 * 4 bytes
        mOptStructures[lvl] = (reinterpret_cast<Vec3f*>(Eigen::internal::aligned_malloc(camMat.area[lvl]*sizeof(Vec3f))));
        I3D_LOG(i3d::detail) << "size: " << camMat.area[lvl]*sizeof(Vec3f) << " lvl: " << lvl;
    }
    I3D_LOG(i3d::info) << "Adding distancetransform for frame: " << std::fixed << mFrameSet->mTimestamp;
    mDtPyramids.push_back(mFrameSet->distanceTransform);
    const float* dtPt = reinterpret_cast<float*>(mFrameSet->distanceTransform.data);
    Vec3f* optLvl0 = mOptStructures.front();
    //Fill the first level of the optimization structure with the DT
    for (auto idx = 0; idx < camMat.area[0]; ++idx) optLvl0[idx][0] = dtPt[idx];
    for (size_t lvl = 0; lvl < PyramidLevels; ++lvl)
    {
        Vec3f *optPyrLvl = mOptStructures[lvl];
        const auto wLvl = camMat.width[lvl];
        const auto hLvl = camMat.height[lvl];
        if (lvl > 0) //down-scaling
        {
            cv::Mat dtLvl(camMat.height[lvl],wLvl,CV_32FC1);
            Eigen::Vector3f *optPyrTopLvl = mOptStructures[lvl-1];
            const auto wTopLvl = camMat.width[lvl-1];
            for (auto col = 0; col < wLvl; ++col)
                for (auto row = 0; row < hLvl; ++row)
                {
                    //to downscale the DT a simple mean is not enough, since the pixel distances also half! -> instead of 4, we divide by 8
                    optPyrLvl[col + row * wLvl][0] = (optPyrTopLvl[2*col + 2*row*wTopLvl][0]+optPyrTopLvl[2*col+1 + 2*row*wTopLvl][0]+
                                                      optPyrTopLvl[2*col+1 + 2*row*wTopLvl+wTopLvl][0]+optPyrTopLvl[2*col + 2*row*wTopLvl+wTopLvl][0]) * 0.125f;
                    dtLvl.at<float>(row,col) = optPyrLvl[col + row * wLvl][0];
                }
            mDtPyramids.push_back(dtLvl);
        }

        //for all the lvls compute the gradient in x and y
        //skip first and last line
        //first and last column are also wrong, but we simply do not edges reprojected there!
        for (auto idx = wLvl; idx < wLvl*(hLvl-1); ++idx)
        {
            //Note: The gradient is inverted compared to photometric approaches!
            optPyrLvl[idx][1] = (optPyrLvl[idx-1][0]-optPyrLvl[idx+1][0]) * 0.5f ; //dx
            optPyrLvl[idx][2] = (optPyrLvl[idx-wLvl][0]-optPyrLvl[idx+wLvl][0]) * 0.5f; //dy
        }
    }

    //Compute on top level, scale down
    mOptStructureAlreadyComputed = true;
}

/**
 * Prepares the frame for tracking.
 * If we estimate the relative pose from keyframe to this frame, we have to compute the DT
 * If we estimate the relative pose from this frame to the keyframe, we only have to compute the valid edges
 */
void FrameData::prepareForTracking(const SystemSettings& settings, const CameraMatrix& camMat)
{
    if (settings.TrackerTrackFromFrameToKf)
        computeValidEdgePixels();
    else
        computeOptimizationStructure(camMat);
}

/**
 * Computes the world pose from the relative pose T_ref_cam and the world pose of the keyframe
 */
const SE3Pose FrameHeader::getWorldPoseFromRef() const 
{ 
    return mReferenceFrame->getPRE_camToWorld()*mT_ref_cam; 
}

void
FrameData::computeValidEdgePixels()
{
    I3D_LOG(i3d::info) << "computingValidEdgePixels for " << std::fixed << mFrameSet->mTimestamp;
    const auto width = mFrameSet->edgeImage.cols, height = mFrameSet->edgeImage.rows;
    const cv::Mat edgeImage = mFrameSet->edgeImage;
    const cv::Mat depthImage = mFrameSet->depth;
    const cv::Mat gradMagn; // -> generate this
    constexpr size_t ApproximateAmountOfEdges{30000};
    mFrameSet->mValidEdgePixels.reserve(ApproximateAmountOfEdges);
    double min,max;
    cv::minMaxIdx(depthImage,&min,&max);
    const auto start = Timer::getTime();
    //Don't go through the border pixels at the moment 
    for (auto xx = 3; xx <= width-4; ++xx)
        for (auto yy = 3; yy <= height-4; ++yy)
        {
            const uint8_t edgePixel = edgeImage.at<uint8_t>(yy,xx);
            if (isEdge(edgePixel))
            {
                const float depthVal = depthImage.at<float>(yy,xx);
                if (isValidDepth(depthVal,mSystemSettings.InputDepthMin,mSystemSettings.InputDepthMax))
                {
                    //Think about computing a weight for the edges!
                    const auto edgeWeight = 1.0f;
                    const Vec3b color = mFrameSet->colorImg.at<Vec3b>(yy,xx);
                    mFrameSet->mValidEdgePixels.push_back(std::make_unique<DetectedEdge>(this,1.0f/depthVal,xx,yy,edgeWeight,color.cast<float>()));
                }
            }
        }
    const auto end = Timer::getTime();
    I3D_LOG(i3d::info) << "Time for edge list computation: " << Timer::getTimeDiffMiS(start,end) << " size: " << mFrameSet->mValidEdgePixels.size();
}

void
FrameData::removeOutOfBounds(const WindowedOptimizer& windowedOptimizer, const CameraMatrix& camMat,
                             const std::vector<FrameData*>& fhsToMargPoints,const SystemSettings& config)
{
    size_t flag_oob = 0, flag_in = 0, flag_inin = 0, flag_nores = 0;
    int idx = 0;
    for (auto& ph : mEdgesGood)
    {
        if(ph == nullptr) continue;
        if(ph->getIdepthScaled() < 0 || ph->edgeResiduals.size()==0) 
        {
            mEdgesOut.push_back(ph);
            I3D_LOG(i3d::detail) << "Dropping: " << ph->efPoint;
            ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
            ph = nullptr;
            flag_nores++;
        }
        else 
        {
            I3D_LOG(i3d::detail) <<"ph->isOOB(fhsToMargPoints): " << ph->isOOB(fhsToMargPoints)
                               << " mFlaggedForMarginalization: " << mFlaggedForMarginalization;
            if(ph->isOOB(fhsToMargPoints) || mFlaggedForMarginalization)
            {
                flag_oob++;
                if(ph->isInlierNew())
                {
                    flag_in++;
                    size_t ngoodRes = 0;
                    for(EdgeFrameResidual* r : ph->edgeResiduals)
                    {
                        r->resetOOB();
                        r->linearize(camMat,config);
                        r->efResidual->isLinearized = false;
                        r->applyRes(true);
                        if(r->efResidual->isActive())
                        {
                            r->efResidual->fixLinearizationF(windowedOptimizer);
                            ngoodRes++;
                        }
                    }
                    if(ph->idepth_hessian > setting_minIdepthH_marg)
                    {
                        flag_inin++;
                        ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
                        mEdgesMarginalized.push_back(ph);
                        I3D_LOG(i3d::debug) << "ph->idepth_hessian: " << ph->idepth_hessian << " > " << setting_minIdepthH_marg;
                    }
                    else
                    {
                        I3D_LOG(i3d::debug) << "Dropping: " << ph->efPoint;
                        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
                        mEdgesOut.push_back(ph);
                    }
                }
                else
                {
                    mEdgesOut.push_back(ph);
                    I3D_LOG(i3d::debug) << "Dropping: " << ph->efPoint;
                    ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
                }

                ph = nullptr;
                I3D_LOG(i3d::debug) << "ph = nullptr: " << ph << " and " << mEdgesGood[idx];
                idx++;
            }
        }

    }
    I3D_LOG(i3d::debug) << "flag_oob: " << flag_oob << " flag_in: " << flag_in << "flag_inin: " << flag_inin << " flag_nores: " << flag_nores;
    size_t nBad = 0;
    for(size_t i=0;i<mEdgesGood.size();i++)
        if (mEdgesGood[i] == nullptr) ++nBad;
    I3D_LOG(i3d::debug) << "size before erase: " << mEdgesGood.size() << "nBad: " << nBad;
    mEdgesGood.erase(std::remove(mEdgesGood.begin(),mEdgesGood.end(),nullptr),mEdgesGood.end());
    I3D_LOG(i3d::debug) << "size after erase: " << mEdgesGood.size() << " total: " << mEdgesGood.size()+nBad;
}

/**
 * Remove edge if they are already in a window
 */
void FrameData::cleanUpPointsAlreadyInWindow()
{
    auto& vEdges = mFrameSet->mValidEdgePixels; //just an alias
    vEdges.erase(std::remove_if(vEdges.begin(),vEdges.end(),[](const auto& e){return e->flagAlreadyInWindow;}),vEdges.end());
}

size_t
FrameData::removeGoodEdgesWithoutResiduals()
{
    //Sort such that the edges without residual are at the back
    auto resIt = std::partition(mEdgesGood.begin(),mEdgesGood.end(),[](const EdgePixel* ph) { return !ph->edgeResiduals.empty(); });
    const auto nRemoved = std::distance(resIt,mEdgesGood.end());

    //Add the removed to edgesOut and set the stateFlag
    std::transform(resIt,mEdgesGood.end(),std::back_inserter(mEdgesOut),[](EdgePixel* ph) 
    { 
        ph->efPoint->stateFlag = EFPointStatus::PS_DROP; 
        I3D_LOG(i3d::debug) << "Deleting: " << ph <<  "ph->efPoint" << ph->efPoint;
        return ph; 
    });
    //Actually erase the bad edges
    mEdgesGood.erase(resIt,mEdgesGood.end());
    return nRemoved;
}

const Vec3f *  
FrameData::getOptStructurePtr(const size_t lvl) const
{
    if (mOptStructures.empty())
    {
        I3D_LOG(i3d::error) << "Optimization structure not computed!";
        exit(0);
    }
    return mOptStructures[lvl];
}


/** 
 * Camera Matrix constructed from settings file
 */
CameraMatrix::CameraMatrix(const SystemSettings& settings)
{
    setLvl0FromParameters(settings.IntrinsicsCamFx,settings.IntrinsicsCamFy,settings.IntrinsicsCamCx,settings.IntrinsicsCamCy,
                          settings.IntrinsicsImgSize.width,settings.IntrinsicsImgSize.height);
}

inline void 
CameraMatrix::setValueScaled(const VecC &value_scaled)
{
    this->value_scaled = value_scaled;
    this->value_scaledf = this->value_scaled.cast<float>();
    value[0] = SCALE_F_INVERSE * value_scaled[0];
    value[1] = SCALE_F_INVERSE * value_scaled[1];
    value[2] = SCALE_C_INVERSE * value_scaled[2];
    value[3] = SCALE_C_INVERSE * value_scaled[3];

    this->value_minus_value_zero = this->value - this->value_zero;
    this->value_scaledi[0] = 1.0f / this->value_scaledf[0];
    this->value_scaledi[1] = 1.0f / this->value_scaledf[1];
    this->value_scaledi[2] = - this->value_scaledf[2] / this->value_scaledf[0];
    this->value_scaledi[3] = - this->value_scaledf[3] / this->value_scaledf[1];
    I3D_LOG(i3d::debug) << "value_scaled: " << value_scaled.transpose()
                        << " value: " << value.transpose()
                        << "value_minus_value_zero: " << value_scaledf.transpose();
}
void 
CameraMatrix::makeK()
{
    fx[0] = fxl();        fy[0] = fyl();
    cx[0] = cxl();        cy[0] = cyl();
    fx_inv[0] = 1.0f/fx[0]; fy_inv[0] = 1.0f/fy[0];
    cx_div_fx = -cx[0]/fx[0]; cy_div_fy = -cy[0]/fy[0]; //the ratio always stays the same!
    K[0] << fx[0], 0, cx[0], 0, fy[0], cy[0], 0, 0, 1;
    K_inv[0] << fx_inv[0],0,cx_div_fx,0,fy_inv[0],cy_div_fy,0,0,1;
    for (size_t lvl = 1; lvl < PyramidLevels; ++lvl)
    {
        const float factor = (1 << lvl);
        const float invFactor = 1.0f/factor;

        width[lvl] = width[lvl-1]*0.5f;
        wBounds[lvl] = width[lvl]-2;
        height[lvl] = height[lvl-1]*0.5f;
        hBounds[lvl] = height[lvl]-2;

        fx[lvl] = fx[0]*invFactor; fy[lvl] = fy[0]*invFactor;
        cx[lvl] = cx[0]*invFactor; cy[lvl] = cy[0]*invFactor;
        fx_inv[lvl] = factor * fx_inv[0]; fy_inv[lvl] = factor * fx_inv[0];
        K[lvl] << fx[lvl],0,cx[lvl],0,fy[lvl],cy[lvl],0,0,1;
        K_inv[lvl] << fx_inv[lvl], 0, cx_div_fx,0,fy_inv[lvl],cy_div_fy,0,0,1;
        area[lvl] = width[lvl]*height[lvl];
        //I3D_LOG(i3d::info) << "K: " << K[lvl] << " lvl: " << lvl << "w/h" << width[lvl] <<"/"<<height[lvl] << "K_inv:"<< K_inv[lvl];
    }
}
void 
CameraMatrix::setLvl0FromParameters(float fx_0, float fy_0, float cx_0, float cy_0, int width_0, int height_0)
{
    VecC initial_value = VecC::Zero();
    initial_value[0] = fx_0;
    initial_value[1] = fy_0;
    initial_value[2] = cx_0;
    initial_value[3] = cy_0;

    setValueScaled(initial_value);
    value_zero = value;
    value_minus_value_zero.setZero();


    width[0] = width_0; wBounds[0] = width_0-2;
    height[0] = height_0; hBounds[0] = height_0-2;
    imageBounds[0] << 1,1,wBounds[0],hBounds[0];
    fx[0] = fx_0; fy[0] = fy_0;
    fx_inv[0] = 1.0f/fx_0; fy_inv[0] = 1.0f/fy_0;
    cx[0] = cx_0; cy[0] = cy_0;
    cx_div_fx = -cx_0/fx_0; cy_div_fy = -cy_0/fy_0; //the ratio always stays the same!
    K[0] << fx_0, 0, cx_0, 0, fy_0, cy_0, 0, 0, 1;
    K_inv[0] << fx_inv[0],0,cx_div_fx,0,fy_inv[0],cy_div_fy,0,0,1;
    area[0] = width[0]*height[0];
    I3D_LOG(i3d::debug) << "K: " << K[0] << " lvl: " << 0<< "w/h" << width[0] <<"/"<<height[0] << "K_inv:"<< K_inv[0];

    for (size_t lvl = 1; lvl < PyramidLevels; ++lvl)
    {
        const float factor = (1 << lvl);
        const float invFactor = 1.0f/factor;

        width[lvl] = width[lvl-1]*0.5f;
        wBounds[lvl] = width[lvl]-2;
        height[lvl] = height[lvl-1]*0.5f;
        hBounds[lvl] = height[lvl]-2;
        imageBounds[lvl] << 1,1,wBounds[lvl],hBounds[lvl];
        fx[lvl] = fx[0]*invFactor; fy[lvl] = fy[0]*invFactor;
        cx[lvl] = cx[0]*invFactor; cy[lvl] = cy[0]*invFactor;
        fx_inv[lvl] = factor * fx_inv[0]; fy_inv[lvl] = factor * fx_inv[0];
        K[lvl] << fx[lvl],0,cx[lvl],0,fy[lvl],cy[lvl],0,0,1;
        K_inv[lvl] << fx_inv[lvl], 0, cx_div_fx,0,fy_inv[lvl],cy_div_fy,0,0,1;
        area[lvl] = width[lvl]*height[lvl];
        //I3D_LOG(i3d::info) << "K: " << K[lvl] << " lvl: " << lvl << "w/h" << width[lvl] <<"/"<<height[lvl] << "K_inv:"<< K_inv[lvl];
    }
}

double
FrameData::computeTargetPrecalcDistScore(const FrameData * const latest) const
{
    double distScore = 0;
    for(const FrameFramePrecalc &ffh : targetPrecalc)
    {
        if(ffh.target->mKeyFrameId + LMS::MinFrameAge-1 > latest->mKeyFrameId/*- LMS::MinFrameAge + 1*/ || ffh.target == ffh.host) continue;
        distScore += 1/(1e-5+ffh.distanceLL);

    }
    distScore *= -std::sqrt(targetPrecalc.back().distanceLL);
    I3D_LOG(i3d::debug) << "fh: " << mKeyFrameId << " frameIdAll: " << mFrameHeader->mFrameId<< "distScore: "
                        << distScore << " sqrtf: " << targetPrecalc.back().distanceLL;
    return distScore;
}


void
FrameData::computeTargetPrecalc(const std::vector<FrameData*>& localWindow, const CameraMatrix& camMat)
{
        targetPrecalc.resize(localWindow.size());
        for (auto i = 0u; i < localWindow.size(); ++i)
        {
            targetPrecalc[i].set(this,localWindow[i],camMat);
        }
}

void 
FrameFramePrecalc::set(FrameData* host, FrameData* target, const CameraMatrix& HCalib )
{
    this->host = host;
    this->target = target;
    const SE3Pose leftToLeft_0 = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
    PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();
    PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();

    const SE3Pose leftToLeft = target->getPRE_worldToCam() * host->getPRE_camToWorld();

    PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
    PRE_tTll = (leftToLeft.translation()).cast<float>();
    distanceLL = leftToLeft.translation().norm();


	Mat33f K = Mat33f::Zero();
    K(0,0) = HCalib.fxl();
    K(1,1) = HCalib.fyl();
    K(0,2) = HCalib.cxl();
    K(1,2) = HCalib.cyl();
    K(2,2) = 1;

    const Mat33f Kinv = K.inverse();
	PRE_KRKiTll = K * PRE_RTll * Kinv;
	PRE_RKiTll = PRE_RTll * Kinv;
	PRE_KtTll = K * PRE_tTll;

}

void 
FrameData::setEvalPT_scaled(const SE3Pose &worldToCam_evalPT)
{
    I3D_LOG(i3d::debug) << mFrameHeader->mFrameId << ": worldToCam_evalPT: " << worldToCam_evalPT.matrix();
    Vec6 initial_state = Vec6::Zero();
    this->worldToCam_evalPT = worldToCam_evalPT;
    setStateScaled(initial_state);
    setStateZero(this->get_state());
}

void 
FrameData::setEvalPT(const SE3Pose &worldToCam_evalPT, const Vec6 &state)
{
    I3D_LOG(i3d::debug) << mFrameHeader->mFrameId << ": worldToCam_evalPT: " << worldToCam_evalPT.matrix() << " state: " << state;
    this->worldToCam_evalPT = worldToCam_evalPT;
    setState(state);
    setStateZero(state);
}

void 
FrameData::setState(const Vec6 &state)
{
    if (FIX_WORLD_POSE) return;
    this->state = state;
    state_scaled.segment<3>(0) = SCALE_XI_TRANS * state.segment<3>(0);
    state_scaled.segment<3>(3) = SCALE_XI_ROT * state.segment<3>(3);
    I3D_LOG(i3d::debug) << "setState in " << mFrameHeader->mFrameId << " state: " << state.transpose();
    setPRE_worldToCam(SE3Pose::exp(w2c_leftEps()) * get_worldToCam_evalPT());
}
void 
FrameData::setStateScaled(const Vec6 &state_scaled)
{
    if (FIX_WORLD_POSE) return;
    this->state_scaled = state_scaled;
    state.segment<3>(0) = SCALE_XI_TRANS_INVERSE * state_scaled.segment<3>(0);
    state.segment<3>(3) = SCALE_XI_ROT_INVERSE * state_scaled.segment<3>(3);
    setPRE_worldToCam(SE3Pose::exp(w2c_leftEps()) * get_worldToCam_evalPT());
}

void 
FrameData::setStateZero(const Vec6 &state_zero)
{
    assert(state_zero.head<6>().squaredNorm() < 1e-20);
    this->state_zero = state_zero;
    for(int i=0;i<6;i++)
    {
        Vec6 eps; eps.setZero(); eps[i] = 1e-3;
        SE3Pose EepsP = SE3Pose::exp(eps);
        SE3Pose EepsM = SE3Pose::exp(-eps);
        SE3Pose w2c_leftEps_P_x0 = (get_worldToCam_evalPT() * EepsP) * get_worldToCam_evalPT().inverse();
        SE3Pose w2c_leftEps_M_x0 = (get_worldToCam_evalPT() * EepsM) * get_worldToCam_evalPT().inverse();
        nullspaces_pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3);
    }

    // scale change
    SE3Pose w2c_leftEps_P_x0 = (get_worldToCam_evalPT());
    w2c_leftEps_P_x0.translation() *= 1.00001;
    w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * get_worldToCam_evalPT().inverse();
    SE3Pose w2c_leftEps_M_x0 = (get_worldToCam_evalPT());
    w2c_leftEps_M_x0.translation() /= 1.00001;
    w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * get_worldToCam_evalPT().inverse();
    nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3);
}

EdgePixel::EdgePixel(const DetectedEdge& detectedEdge)
    :host(detectedEdge.host),hostX(detectedEdge.hostX),hostY(detectedEdge.hostY),color(detectedEdge.color)
{
    constexpr float settings_initDepthWeight{2};
    weight = settings_initDepthWeight;
    idepth_init = detectedEdge.idepth;
    setIdepthScaled(idepth_init);
    setIdepthZero(idepth);
    
    idepth_hessian = 0;
    hasDepthPrior = true; 
    numGoodResiduals= 0;
    maxRelBaseline = 0;
    status = INACTIVE;
    efPoint = nullptr;
    
    if (std::isnan(idepth))
    {
        I3D_LOG(i3d::info) << "invDepth is nan!";
    }
    assert(!std::isnan(idepth));
}

void 
FrameData::findResidualsToOptimize(DetectedEdgesPtrVec& newEdgePixels, CoarseDistanceMap * const coarseDistanceMap,
                                   const FrameData* const refFrameNew, const CameraMatrix& cameraMat, const SystemSettings& config,
                                   const float compareDist) const
{
    LOG_THRESHOLD(i3d::nothing);
    I3D_LOG(i3d::detail) << "kfid: " << mKeyFrameId << ":  " <<  refFrameNew->PRE_worldToCam.matrix3x4() << " " << PRE_camToWorld.matrix3x4()
                       << "newEdgePixels.size(): " << newEdgePixels.size();
    assert(refFrameNew->mFrameHeader->mFrameId != mFrameHeader->mFrameId && "Ids shouldn't be equal!");
    constexpr size_t EvalLvlDistMap{1};
    const SE3Pose oldToNew = refFrameNew->PRE_worldToCam * PRE_camToWorld;
    
    const Mat33f KRKi = cameraMat.K[EvalLvlDistMap]*oldToNew.rotationMatrix().cast<float>()*cameraMat.K_inv[0];
    const Vec3f Kt = cameraMat.K[EvalLvlDistMap]*oldToNew.translation().cast<float>(); //very similar to DSO
    const Vec3f* distTrans = refFrameNew->getOptStructurePtr(EvalLvlDistMap);
    size_t nInvEdges = 0, nValidEdges = 0, nInvOOB{0}, nInvDist{0},nInvForOpt{0};
    auto& validEdges = mFrameSet->mValidEdgePixels;
    const int width = cameraMat.width[EvalLvlDistMap];//, hG = cameraMat.height[EvalLvlDistMap];
    cv::Mat dbgImg = returnRgbFullSize().clone();
    I3D_LOG(i3d::detail) << "validEdges " << validEdges.size();
    // CHANGE VALID EDGES TO POINTER!!
    for (const auto& validEdge : validEdges)
    {
        if (validEdge->isValidForOptimizing())
        {
            //Now, project the point from host and to current frame, normalize and check bounds
            const Vec2f pt2D = validEdge->reproject(KRKi,Kt);//(KRKi*validEdge->getPixel2DHom()+Kt*validEdge->idepth).hnormalized();
            if (isInImage(width,cameraMat.height[EvalLvlDistMap],pt2D[0],pt2D[1]))
            {
                const float r = getInterpolatedElement31(distTrans,pt2D[0],pt2D[1],width);
                const int u = int(pt2D[0]),v = int(pt2D[1]);
                
                const float dist = (coarseDistanceMap == nullptr ? 1000 :
                                    coarseDistanceMap->fwdWarpedIDDistFinal[u+width*v] + (pt2D[0]-std::floor((float)(pt2D[0]))));

                if (r < mSystemSettings.LocalMapperMaxDistForValidPixels && dist >= compareDist) // && dist >= config.LocalMapperMinDistInDistMapForValidPixels) //edge must be really close!
                {
                    
                    if (coarseDistanceMap != nullptr) 
                    {
                        coarseDistanceMap->addIntoDistFinal(u,v);
                        I3D_LOG(i3d::detail) << "Adding: " << u << ", " << v << " to dist map for pt: " << validEdge->getPixel2D().transpose() << " dist: " << dist
                                            << "validEdge: " << validEdge.get();
                        cv::circle(dbgImg,cv::Point2i(u,v),2,cv::Vec3b(0,255,0));
                    }
                    newEdgePixels.push_back(validEdge.get());
                    I3D_LOG(i3d::detail) << "Added valid edge: " << newEdgePixels.back();
                    ++nValidEdges;
                }
                else 
                {
                    I3D_LOG(i3d::detail) << "invalidEdge: " << validEdge->getPixel2D().transpose() << " dist: " << dist << " r: " << r
                                        << "validEdge: " << validEdge.get();
                    ++nInvDist;
                }
            }
            else 
            {
                ++nInvOOB; 
                validEdge->flagEdgeOOB = true;
                I3D_LOG(i3d::detail) << "invalidEdge oob: " << validEdge->getPixel2D().transpose() << " pt2D: " << pt2D.transpose();
            }
        }
        else 
        {
            ++nInvForOpt;
            exit(0);
        }
        ++nInvEdges;
    }
    // cv::Mat fwdDist(cameraMat.height[1],cameraMat.width[1],CV_32FC1,coarseDistanceMap->fwdWarpedIDDistFinal);
    // // double min,max;
    // // cv::minMaxIdx(fwdDist,&min,&max);
    // cv::imshow("fwdDist",fwdDist/15.0);
    // cv::imshow("dbgAct",dbgImg);
    // cv::waitKey(0);
    I3D_LOG(i3d::detail) << "validEdges before erase: " << validEdges.size();
    //TODO: CHECK IF WE SHOULD REALLY DELETE HERE! 
    //TODO: Delete OOB pixels
    validEdges.erase(std::remove_if(std::begin(validEdges),std::end(validEdges),
                                    [](const auto& edge){ 
                                        // if (edge->flagEdgeOOB){I3D_LOG(i3d::info) << "Deleting: " << edge->getPixel2D().transpose();}
                                        return edge->flagEdgeOOB; }),
                                    std::end(validEdges));
    I3D_LOG(i3d::detail)<< "validEdges after erase: " << validEdges.size()
                        << "Valid edges: " << nValidEdges 
                        << " Invalid edges: " << nInvEdges-nValidEdges
                        << " because of OOB: " << nInvOOB
                        << " because of dist: " << nInvDist
                        << " because of isValid: " << nInvForOpt
                        << "newEdgePixels.size():" << newEdgePixels.size();
}

double 
EdgeFrameResidual::linearize(const CameraMatrix &cameraMat, const SystemSettings& config)
{
    state_NewEnergyWithOutlier = -1;
    if(state_state == ResState::OOB)
    {
        I3D_LOG(i3d::debug) << "Setting to OOB because it was previously OOB!";
        state_NewState = ResState::OOB;
        return state_energy;
    }

    const FrameFramePrecalc& precalc = host->getTargetPrecalcById(target->idx);
    float energyLeft = 0;
    const Eigen::Vector3f* dIl = target->getOptStructurePtr(0);//getOptPyrLvl(0);
    const Mat33f &PRE_KRKiTll = precalc.PRE_KRKiTll;
    const Vec3f &PRE_KtTll = precalc.PRE_KtTll;
    const Mat33f &PRE_RTll_0 = precalc.PRE_RTll_0;
    const Vec3f &PRE_tTll_0 = precalc.PRE_tTll_0;

    Vec6f d_xi_x, d_xi_y;   ///< derivatives of the rotation and translation
    Vec4f d_C_x, d_C_y;     ///< camera paramters
    float d_d_x, d_d_y;     ///< inverse depth derivatives

    float drescale, u, v, new_idepth;
    float Ku, Kv;
    Vec3f KliP;
    Eigen::Vector2f pt2DnoK,pt2D;
    if (!projectPixel(edgePixel->hostX,edgePixel->hostY,
                        cameraMat,edgePixel->getIdepthZeroScaled(),
                        PRE_RTll_0,PRE_tTll_0,
                        new_idepth,pt2DnoK,pt2D,KliP,drescale))
    {
        I3D_LOG(i3d::debug) << "Setting to OOB because projectPixel is OOB!" << pt2D.transpose() << " new_idepth= " << new_idepth << " vs old : " << edgePixel->getIdepthZeroScaled();
        state_NewState = ResState::OOB;
        return state_energy;
    }

    const Vec2f pt2D_2 = (PRE_KRKiTll * Vec3f(edgePixel->hostX,edgePixel->hostY, 1) + PRE_KtTll*edgePixel->getIdepthScaled()).hnormalized();
    Vec3f resInterp(0,0,0);
    if (isInImageGreater(cameraMat.imageBounds[0],pt2D_2))
    {
        resInterp = getInterpolatedElement33(dIl, pt2D_2[0], pt2D_2[1], cameraMat.width[0]);
    }
    else
    {
        I3D_LOG(i3d::debug) << "Setting to OOB because projection with PRE_KRKiTll is OOB!" << pt2D.transpose();
        state_NewState = ResState::OOB;
        return state_energy;
    }
    auto residual = resInterp[0];
    if (residual > config.LocalMapperEdgeDistance)//settings_edgeFilterThreshold) //outlier
    {
        I3D_LOG(i3d::debug) << "Setting to OOB because residual > 10!";
        state_NewState = ResState::OUTLIER;
        return state_energy;
    }
    const auto hw = getWeightOfEvoR(residual,config.TrackerResidualHuberWeight);


    u = pt2DnoK[0]; v = pt2DnoK[1];
    Ku = pt2D[0]; Kv = pt2D[1];

    centerProjectedTo = Vec3f(Ku, Kv, new_idepth);

    // diff d_idepth
    //this is simply the derivative with respect to invDepth!
    //fx*(tx - tz*x)*(r22 + invD*tz + (r02*x)/fx + (r12*y)/fy - (cx*r02)/fx - (cy*r12)/fy)
    d_d_x = drescale * (PRE_tTll_0[0]-PRE_tTll_0[2]*u)*SCALE_IDEPTH*cameraMat.fxl();
    //fy*(ty - tz*y)*(r22 + invD*tz + (r02*x)/fx + (r12*y)/fy - (cx*r02)/fx - (cy*r12)/fy)
    d_d_y = drescale * (PRE_tTll_0[1]-PRE_tTll_0[2]*v)*SCALE_IDEPTH*cameraMat.fyl();

    // diff calib
    //-(r00 - r02*x)/(r22 + invD*tz + (r02*x)/fx + (r12*y)/fy - (cx*r02)/fx - (cy*r12)/fy)
    d_C_x[2] = drescale*(PRE_RTll_0(2,0)*u-PRE_RTll_0(0,0));
    d_C_x[3] = cameraMat.fxl() * drescale*(PRE_RTll_0(2,1)*u-PRE_RTll_0(0,1)) * cameraMat.fyli();
    d_C_x[0] = KliP[0]*d_C_x[2]; //==((r00 - r02*x)*(cx - x)*(r22 + invD*tz + (r02*x)/fx + (r12*y)/fy - (cx*r02)/fx - (cy*r12)/fy))/fx
    d_C_x[1] = KliP[1]*d_C_x[3];

    d_C_y[2] = cameraMat.fyl() * drescale*(PRE_RTll_0(2,0)*v-PRE_RTll_0(1,0)) * cameraMat.fxli();
    d_C_y[3] = drescale*(PRE_RTll_0(2,1)*v-PRE_RTll_0(1,1));
    d_C_y[0] = KliP[0]*d_C_y[2]; //1 - (r00 - r02*x)*(r22 + invD*tz + (r02*x)/fx + (r12*y)/fy - (cx*r02)/fx - (cy*r12)/fy)
    d_C_y[1] = KliP[1]*d_C_y[3];

    d_C_x[0] = (d_C_x[0]+u)*SCALE_F;
    d_C_x[1] *= SCALE_F;
    d_C_x[2] = (d_C_x[2]+1)*SCALE_C; //
    d_C_x[3] *= SCALE_C;

    d_C_y[0] *= SCALE_F;
    d_C_y[1] = (d_C_y[1]+v)*SCALE_F;
    d_C_y[2] *= SCALE_C;
    d_C_y[3] = (d_C_y[3]+1)*SCALE_C;

    //this looks like the projection Jacobian
    d_xi_x[0] = new_idepth*cameraMat.fxl();
    d_xi_x[1] = 0;
    d_xi_x[2] = -new_idepth*u*cameraMat.fxl(); //should be u*fx/Z^2
    d_xi_x[3] = -u*v*cameraMat.fxl();
    d_xi_x[4] = (1+u*u)*cameraMat.fxl();
    d_xi_x[5] = -v*cameraMat.fxl();

    d_xi_y[0] = 0;
    d_xi_y[1] = new_idepth*cameraMat.fyl();
    d_xi_y[2] = -new_idepth*v*cameraMat.fyl();//should be v*fy/Z^2
    d_xi_y[3] = -(1+v*v)*cameraMat.fyl();
    d_xi_y[4] = u*v*cameraMat.fyl();
    d_xi_y[5] = u*cameraMat.fyl();

    {

        J->Jpdxi[0] = d_xi_x; //ok
        J->Jpdxi[1] = d_xi_y; //ok
        if (edgePixel->host->FIX_WORLD_POSE)
        {
            J->Jpdxi[0].setZero();
            J->Jpdxi[1].setZero();
        }
        // I3D_LOG(i3d::info) << edgePixel->host->mKeyFrameId << "J->Jpdxi[0]: " << J->Jpdxi[0].transpose() << "  J->Jpdxi[1]: " << J->Jpdxi[1].transpose();
        //also ok
        J->Jpdc[0] = d_C_x; //x + ((r00 - r02*x)*(cx - x)*(r22 + invD*tz + (r02*x)/fx + (r12*y)/fy - (cx*r02)/fx - (cy*r12)/fy))/fx
        J->Jpdc[1] = d_C_y;
        //DO NOT OPTIMIZE DEPTH AT THE MOMENT!

        J->Jpdd[0] = d_d_x; //ok
        J->Jpdd[1] = d_d_y; //ok
        //Now, check if should start optimizing depth
        const auto d_idepth = resInterp.tail<2>().dot(J->Jpdd);
        if (d_idepth*d_idepth*hw > setting_minIdepthH_act)
        {
            // I3D_LOG(i3d::info) << "Switching back to optimiziation: " << d_idepth*d_idepth*hw << "<" << setting_minIdepthH_act;
            FLAG_DONT_OPT_DEPTH = false;
        }
        if (FLAG_DONT_OPT_DEPTH) J->Jpdd.setZero();
    }

    J->resF = residual*hw;
    energyLeft = residual*J->resF; //== res*res*hw
    resInterp = resInterp*(-1);
    resInterp[1] *= hw;
    resInterp[2] *= hw;
    J->JIdx[0] = /*hw**/resInterp[1];
    J->JIdx[1] = /*hw**/resInterp[2];
//    auto wJI2_sum = hw*(resInterp[1]*resInterp[1]+resInterp[2]*resInterp[2]);


    const float JIdxJIdx_00=resInterp[1]*resInterp[1];
    const float JIdxJIdx_11=resInterp[2]*resInterp[2];
    const float JIdxJIdx_10=resInterp[1]*resInterp[2];

    //inner product -> approximation of the Hessian
    //Check, if we really need that
    //Photometric derivatives are not needed!
    J->JIdx2(0,0) = JIdxJIdx_00;
    J->JIdx2(0,1) = JIdxJIdx_10;
    J->JIdx2(1,0) = JIdxJIdx_10;
    J->JIdx2(1,1) = JIdxJIdx_11;
    state_NewEnergyWithOutlier = energyLeft;
    const auto settings_edgeFilterThreshold = config.LocalMapperEdgeDistance;
    if (energyLeft > settings_edgeFilterThreshold*settings_edgeFilterThreshold*getWeightOfEvoR(settings_edgeFilterThreshold,config.TrackerResidualHuberWeight))// || wJI2_sum < 2)
    {
        I3D_LOG(i3d::debug) << "Setting to OUTLIER because energy > 30*30*weight" << energyLeft << " vs " << settings_edgeFilterThreshold*settings_edgeFilterThreshold*getWeightOfEvoR(settings_edgeFilterThreshold,config.TrackerResidualHuberWeight);
        energyLeft = settings_edgeFilterThreshold*settings_edgeFilterThreshold*getWeightOfEvoR(settings_edgeFilterThreshold,config.TrackerResidualHuberWeight);
        state_NewState = ResState::OUTLIER;
    }
    else
    {
        state_NewState = ResState::IN;
    }
    state_NewEnergy = energyLeft;
    return energyLeft;
}

void 
EdgeFrameResidual::applyRes(bool copyJacobians)
{
    if(copyJacobians)
    {
        if(state_state == ResState::OOB)
        {
            assert(!efResidual->isActiveAndIsGoodNEW);
            return;	// can never go back from OOB
        }
        if(state_NewState == ResState::IN)// && )
        {
            efResidual->isActiveAndIsGoodNEW=true;
            efResidual->takeDataF();
        }
        else
        {
            efResidual->isActiveAndIsGoodNEW=false;
        }
    }

    setState(state_NewState);
    state_energy = state_NewEnergy;
}

void
FrameData::release()
{
    auto freeEdgeList = [](EdgePixelVec& list)
                        {
                            for (auto* elem : list)
                            {
                                elem->release();
                                delete elem;
                                elem = nullptr;
                            }
                        };
    freeEdgeList(mEdgesGood);
    freeEdgeList(mEdgesMarginalized);
    freeEdgeList(mEdgesOut);
    mEdgesGood.clear();
    mEdgesMarginalized.clear();
    mEdgesOut.clear();
    mFrameSet->mValidEdgePixels.clear();
    mFrameSet->depthJet.release();
}

void
FrameData::resetFrame()
{
    //Delete everything that is not required anymore
    release();
}

double 
DetectedEdge::linearizeResidual(const CameraMatrix&  HCalib, const float outlierTHSlack,
                                ImmatureEdgeTemporaryResidual* tmpRes,
                                float &Hdd, float &bd,
                                float idepth,const SystemSettings& config) const
{

    if(tmpRes->state_state == ResState::OOB)
        { tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy; }


    
    FrameFramePrecalc& precalc = host->getTargetPrecalcById(tmpRes->target->idx);

    //F: acceleration structure with intensity,dxI,dyI
    const Vec3f* optDt = tmpRes->target->getOptStructurePtr(0);//tmpRes->target->dI;
//  defintions:
//  SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
//	PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
//	PRE_tTll = (leftToLeft.translation()).cast<float>();
    const Mat33f &PRE_RTll = precalc.PRE_RTll;
    const Vec3f &PRE_tTll = precalc.PRE_tTll;
    //we now assume that we have a new camera matrix Hcalib

//    KliP = Vec3f(
//            (u_pt+dx-HCalib->cxl())*HCalib->fxli(),
//            (v_pt+dy-HCalib->cyl())*HCalib->fyli(),
//            1);

//    Vec3f ptp = R * KliP + t*idepth;

    Eigen::Vector3f newPt3D = PRE_RTll*Vec3f((hostX-HCalib.cxl())*HCalib.fxli(),
                                             (hostY-HCalib.cyl())*HCalib.fyli(),1)+PRE_tTll*idepth;
    const auto drescale = 1.0f/newPt3D[2];
    const auto new_idepth = idepth*drescale;
    if (!(drescale > 0))
    {
        tmpRes->state_NewState = ResState::OOB;
        return tmpRes->state_energy;
    }

    Eigen::Vector2f pt2DnoK(newPt3D[0]*drescale,newPt3D[1]*drescale); //u,v
    Eigen::Vector2f pt2D(pt2DnoK[0]*HCalib.fxl()+HCalib.cxl(),pt2DnoK[1]*HCalib.fyl()+HCalib.cyl()); //Ku,Kv
    
    if (!isInImageGreater(HCalib.imageBounds[0],pt2D) || new_idepth <= 0)
    {
        tmpRes->state_NewState = ResState::OOB;
        return tmpRes->state_energy;
    }


    //Now, optimize
    //Now, we have to interpolate
    const Eigen::Vector3f resInterp =  getInterpolatedElement33(optDt,pt2D[0],pt2D[1],HCalib.width[0]); //residual,dx,dy
    const float residual = resInterp[0];
    if ((std::isnan(resInterp[0]) || std::isnan(resInterp[1]) || std::isnan(resInterp[2])))// || resInterp[0] > 30)
    {
        tmpRes->state_NewState = ResState::OOB;
        return tmpRes->state_energy;
    }
    if (resInterp[0] > config.LocalMapperMaxDistForValidPixels)
    {
        tmpRes->state_NewState = ResState::OUTLIER;
        // tmpRes->state_NewEnergy = config.LocalMapperMaxDistForValidPixels*config.TrackerResidualHuberWeight; //because r*r*t_huber/|r| = r*t_huber
        //FIXME: 30 should be config.LocalMapperMaxDistForValidPixels*config.LocalMapperMaxDistForValidPixels
        tmpRes->state_NewEnergy = 30*config.TrackerResidualHuberWeight;
        return tmpRes->state_NewEnergy;
    }
    //check distance
    const auto resWeight = getWeightOfEvoR(residual, config.TrackerResidualHuberWeight);
    //Compute the derivatives!
    //This might be a bit tricky but we'll see
    const auto resSquared = residual*residual;
    auto energyLeft = resWeight*resSquared;
    const float dxInterp = resInterp[1]*HCalib.fxl(), dyInterp = resInterp[2]*HCalib.fyl();
    const float d_idepth = dxInterp*drescale*(PRE_tTll[0]-PRE_tTll[2]*pt2DnoK[0])+dyInterp*drescale*(PRE_tTll[1]-PRE_tTll[2]*pt2DnoK[1]);

    Hdd = resWeight*d_idepth*d_idepth;
    bd = resWeight*residual*d_idepth;
    tmpRes->state_NewState = ResState::IN;
    tmpRes->state_NewEnergy = energyLeft;
    return energyLeft;
}

bool 
EdgePixel::isInlierNew() const
{
    return edgeResiduals.size() >= setting_minGoodActiveResForMarg && numGoodResiduals >= setting_minGoodResForMarg;
}

bool 
EdgePixel::isOOB(const std::vector<FrameData*>& toMarg) const
{
    size_t visInToMarg{0};
    for(EdgeFrameResidual* r : edgeResiduals)
    {
        if(r->state_state == ResState::IN)
        {
            for(FrameData* k : toMarg)
                if(r->target == k) visInToMarg++;
        }
    }
    if(edgeResiduals.size() >= setting_minGoodActiveResForMarg && numGoodResiduals > setting_minGoodResForMarg+10 &&
        edgeResiduals.size()-visInToMarg < setting_minGoodActiveResForMarg)
    {
        I3D_LOG(i3d::debug) << "Marginalizing: " << edgeResiduals.size() << ">= " << setting_minGoodActiveResForMarg
                            << "numGoodR: " << numGoodResiduals << ">"<<setting_minGoodResForMarg+10
                            << "edgeR.size()-visInToMarg: " << edgeResiduals.size()-visInToMarg << " < " << setting_minGoodResForMarg;
        return true;
    }

    if(lastResiduals[0].second == ResState::OOB) 
    {
        I3D_LOG(i3d::debug) << this << ": lastResiduals[0].second == ResState::OOB";
        return true;
    }
    if(edgeResiduals.size() < 2) return false;
    if(lastResiduals[0].second == ResState::OUTLIER && lastResiduals[1].second == ResState::OUTLIER)
    {
        I3D_LOG(i3d::debug) << this << ": lastResiduals[0].second == ResState::OUTLIER && lastResiduals[1].second == ResState::OUTLIER";
        return true;
    }
    return false;
}

void
EdgePixel::release()
{
    for(auto i = 0u; i < edgeResiduals.size(); i++) delete edgeResiduals[i];
    edgeResiduals.clear();
}
}
