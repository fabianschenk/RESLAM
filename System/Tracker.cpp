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
#include <opencv2/highgui.hpp>

#include "Tracker.h"
#include "../Utils/Logging.h"
#include "../config/Defines.h"
#include "../Utils/utility_functions.h"
#include "../Utils/timer.h"
#include "../Utils/LGSX.h"
#include "../System/SystemSettings.h"
namespace RESLAM
{
Tracker::Tracker(const SystemSettings& systemSettings):
    mSystemSettings(systemSettings), mCameraMatrix(systemSettings),
    mCountingMap(mCameraMatrix.height[0],mCameraMatrix.width[0],CV_8UC1,cv::Scalar(0)),
    mCountingMapAll(mCameraMatrix.height[0],mCameraMatrix.width[0],CV_8UC1,cv::Scalar(0))
{
    I3D_LOG(i3d::info) << "mCameraMatrix: " << mCameraMatrix.fxl() << " " << mCameraMatrix.fyl();
    size_t memSize = mCameraMatrix.area[0] * sizeof(float);//settings.nCamsToOptimize * 2;
    buf_warped_residual = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
    buf_warped_dx = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
    buf_warped_dy = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
    buf_warped_x = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
    buf_warped_y = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
    buf_warped_idepth = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
    buf_warped_weight = static_cast<float*>(Eigen::internal::aligned_malloc(memSize));
}

Tracker::~Tracker()
{
    Eigen::internal::aligned_free((void*)buf_warped_residual);
    Eigen::internal::aligned_free((void*)buf_warped_dx);
    Eigen::internal::aligned_free((void*)buf_warped_dy);
    Eigen::internal::aligned_free((void*)buf_warped_x);
    Eigen::internal::aligned_free((void*)buf_warped_y);
    Eigen::internal::aligned_free((void*)buf_warped_idepth);
    Eigen::internal::aligned_free((void*)buf_warped_weight);  
}

void printm128(const __m128& t1, const std::string& txt)
{
    //return;
    I3D_LOG(i3d::info) << std::fixed << txt << ": " << t1[0] <<", "<< t1[1]<< ", " << t1[2] << ", " << t1[3];
}

/**
 * Tracks from source to targetFrame across all lvls
 */
Tracker::TrackerStatus Tracker::trackFrames(const FrameData& targetFrame, const FrameData& sourceFrame, SE3Pose& T_t_s, ResidualInfo& resInfo) const
{
    LOG_THRESHOLD(i3d::info);
    const SE3Pose backupPose = T_t_s;
    TrackerStatus trackingStatus = TrackerStatus::Lost;
    I3D_LOG(i3d::info) << " initPose: " << T_t_s.matrix3x4();
    for (auto lvl = static_cast<int>(PyramidLevels-1); lvl >= 0; --lvl)
    {
        auto startTime = Timer::getTime();
        I3D_LOG(i3d::info) << "Tracking level: " << lvl << " from " << sourceFrame.mKeyFrameId << " to " << targetFrame.mKeyFrameId;
        trackingStatus = trackFrameOnLvl(targetFrame,sourceFrame,static_cast<size_t>(lvl), T_t_s, resInfo);

        //Something went wrong!
        if (trackingStatus != TrackerStatus::Ok)
        {
            I3D_LOG(i3d::info) << "Problems with tracking on lvl: " << lvl << ": " << printTrackerStatus(trackingStatus);
            //Try again with backup pose
            T_t_s = backupPose;
        }
        if (mSystemSettings.TrackerShowIterationsDebug || flagIterationsDebug)
        {
            reprojectToImage(targetFrame,sourceFrame,T_t_s);
        }
        auto endTime = Timer::getTime();
        I3D_LOG(i3d::info) << "Time for tracking at lvl: " << lvl <<":"<< Timer::getTimeDiffMiS(startTime,endTime) << "/" << Timer::getTimeDiffMs(startTime,endTime);
    }

    //Compute optical flow
    computeOpticalFlow(targetFrame,sourceFrame,T_t_s,resInfo);
    //try new quality assessment
    // auto valid = histogramVotingAfterTracking(targetFrame,sourceFrame,T_t_s,resInfo);
    // I3D_LOG(i3d::info) << "histogramVotingAfterTracking: " << valid;
    LOG_THRESHOLD(i3d::info);
    return trackingStatus;//(valid ? trackingStatus : TrackerStatus::Lost);
}

void
Tracker::computeOpticalFlow(const FrameData& targetFrame, const FrameData& sourceFrame, SE3Pose& T_ref_N, ResidualInfo& resInfo) const
{
    constexpr size_t lvl{0};
    const Mat33f K_inv = mCameraMatrix.K_inv[0];
    const Vec3f t = T_ref_N.translation().cast<float>();
    const Mat33f KRK_i = mCameraMatrix.K[lvl]*T_ref_N.rotationMatrix().cast<float>()*K_inv;
    const Vec3f Kt = mCameraMatrix.K[lvl]*t;
    const size_t width = mCameraMatrix.width[lvl];
    const Vec3f * const optLvl = targetFrame.getOptStructurePtr(lvl);
    I3D_LOG(i3d::detail) << "reprojBoundariesLvl: " << mCameraMatrix.imageBounds[lvl].transpose();
    for (const auto& edgePixel : sourceFrame.mFrameSet->mValidEdgePixels)
    {
        const Vec2f pt2D = edgePixel->getPixel2D();//Vec2E(edgePixel.hostX,edgePixel.hostY);
        const Vec3f ptHom = pt2D.homogeneous();
        const Vec2f pt2DRT = edgePixel->reproject(KRK_i,Kt);
        if (!isInImageGreater(mCameraMatrix.imageBounds[lvl],pt2DRT))/// || new_idepth <= 0)
        {
            continue;
        }

        //Now, we have to interpolate
        const float residual =  getInterpolatedElement31(optLvl,pt2DRT[0],pt2DRT[1],width); //residual,dx,dy
        //check distance
        if (residual >= mSystemSettings.TrackerEdgeDistanceFilter[lvl] && mSystemSettings.TrackerUseEdgeFilter)
        {
            continue;
        }
                //optical flow computation
        resInfo.opticalFlowRT += (pt2DRT-pt2D).norm();
        //== (pt + Kt * d).hormalize()
        const Vec2f pt2DT = (ptHom+ Kt*edgePixel->idepth).hnormalized();
        resInfo.opticalFlowT += (pt2DT-pt2D).norm();
    }
}

Tracker::TrackerStatus 
Tracker::findInitAndTrackFrames(const FrameData& targetFrame, const FrameData& sourceFrame, SE3Pose& T_ref_N, const PoseVector& posesToTry) const
{
    ResidualInfo resInfo;
    SE3Pose initPose;
    if (findInit(targetFrame, sourceFrame, posesToTry, initPose))
    {
        auto trackingStatus = trackFrames(targetFrame,sourceFrame,initPose,resInfo);
        if (trackingStatus != TrackerStatus::Ok) 
        {
            I3D_LOG(i3d::info) << "Tracking lost!";
            return TrackerStatus::Lost;
        }
        T_ref_N = initPose;
        if (posesToTry.size() == 1) 
        {
            I3D_LOG(i3d::info) << "New Keyframe because poses: " << posesToTry.size();
            return TrackerStatus::NewKeyframe; //create a keyframefor the first two frames!
        }
        return analyseTrackingResult(trackingStatus,resInfo);
    }
    return TrackerStatus::Lost;
}

void
Tracker::reprojectToImage(const FrameData& targetFrame, const FrameData& sourceFrame, const SE3Pose& T_t_s, const std::string& title, const bool WaitFlag) const
{
    cv::Mat debugImg;
    std::vector<cv::Mat> mergeImg;
    mergeImg.push_back(targetFrame.mFrameSet->edgeImage);
    mergeImg.push_back(targetFrame.mFrameSet->edgeImage);
    mergeImg.push_back(targetFrame.mFrameSet->edgeImage);
    cv::merge(mergeImg,debugImg);
    const auto KRK_i = mCameraMatrix.K[0]*T_t_s.rotationMatrix().cast<float>()*mCameraMatrix.K_inv[0];
    const auto Kt = mCameraMatrix.K[0]*T_t_s.translation().cast<float>();
    const Vec3b ColorGreen(0,255,0), ColorRed(0,0,255);
    const Vec3f * const optLvl = targetFrame.getOptStructurePtr(0);
    I3D_LOG(i3d::info) << "getEdgesGood.size(): " << sourceFrame.getEdgesGood().size() << " dete: " << sourceFrame.getDetectedEdges().size()
                       << " kf timestamp: " << sourceFrame.mKeyFrameId << " xx " << std::fixed << sourceFrame.mFrameSet->mTimestamp;
    for (const auto& edgePixel : sourceFrame.getDetectedEdges())
    {
        
        const Vec2f pt2DE = edgePixel->reproject(KRK_i,Kt);
        const Vec2i pt2D = pt2DE.cast<int>();
        I3D_LOG(i3d::detail) << "debugImg.cols: " << debugImg.cols << " debugImg.rows: " << debugImg.rows << "optLvl: " << optLvl;
        if (pt2D[0] > 0 && pt2D[1] > 0 && pt2D[0] < debugImg.cols-1 && pt2D[1] < debugImg.rows-1)
        {
            const auto idx = &edgePixel-&sourceFrame.getDetectedEdges()[0];
            I3D_LOG(i3d::detail) << "pt2D: " << pt2D.transpose() << " w: " << mCameraMatrix.width[0] << " optLvl: " << optLvl << " idx: "<< idx << "/"<< sourceFrame.getDetectedEdges().size();;
            const float res = getInterpolatedElement31(optLvl,pt2D[0],pt2D[1],mCameraMatrix.width[0]);
            I3D_LOG(i3d::detail) << "res " << res << " mSystemSettings.TrackerEdgeDistanceFilter[0]: " << mSystemSettings.TrackerEdgeDistanceFilter[0];
            if (res < mSystemSettings.TrackerEdgeDistanceFilter[0])
                debugImg.at<Vec3b>(pt2D[1],pt2D[0]) = ColorGreen;
            else
                debugImg.at<Vec3b>(pt2D[1],pt2D[0]) = ColorRed;
        }
    }
    cv::imshow(title,debugImg);
    if (WaitFlag) cv::waitKey(0);
}

float
Tracker::findInit(const FrameData& targetFrame, const FrameData&  sourceFrame, const PoseVector posesToTry, SE3Pose& T_init) const
{
    auto minCost = FLOAT_INF;
    //What happens when I only have one or two poses?
    for (const SE3Pose& pose : posesToTry)
    {
        const auto testCost = evaluateCostFunction(targetFrame,sourceFrame,pose);
        if (mSystemSettings.TrackerShowInitDebug)
            reprojectToImage(targetFrame,sourceFrame,pose,"findInit");
        if (testCost < minCost)
        {
            minCost = testCost;
            T_init = pose;
        }
    }
    return minCost;
}

float
Tracker::evaluateCostFunction(const FrameData& targetFrame, const FrameData& sourceFrame, const SE3Pose& T_t_s) const
{
    ResidualInfo resInfo;
    constexpr bool FillBuffers{false};
    calculateErrorsAndBuffers(targetFrame,sourceFrame,T_t_s,mSystemSettings.TrackerEvalLvlForInit,resInfo,FillBuffers);
    return resInfo.meanErrorGood(mSystemSettings.TrackerUseWeightsResidualsForErrorComputation);
}

void
Tracker::evalReprojection(const Mat33f& KRK_i, const Vec3f& Kt, const Vec2f& pt2D, const float idepth,
                          const Vec3f * const optLvl, const size_t evalLvl, ResidualInfo& resInfo) const
{
    const Vec2f pt2Dnew =  (KRK_i*pt2D.homogeneous()+Kt*idepth).hnormalized();
    if (!isInImageGreater(mCameraMatrix.imageBounds[evalLvl],pt2Dnew))// || new_idepth <= 0)
    {
        ++resInfo.nBadEdges;
        I3D_LOG(i3d::detail) << "mCameraMatrix.imageBounds[evalLvl]: " << mCameraMatrix.imageBounds[evalLvl].transpose() << " pt2Dnew: " << pt2Dnew.transpose() 
                             << " resInfo: " << resInfo.nBadEdges << " good: " << resInfo.nGoodEdges;
        return;
    }
    const float residual = getInterpolatedElement31(optLvl,pt2Dnew[0],pt2Dnew[1],mCameraMatrix.width[evalLvl]); //residual,dx,dy
    //check distance
    if (residual >= mSystemSettings.TrackerEdgeDistanceFilter[evalLvl] && mSystemSettings.TrackerUseEdgeFilter)
    {
        ++resInfo.nBadEdges;
        I3D_LOG(i3d::detail) << "mSystemSettings.TrackerEdgeDistanceFilter[evalLvl]: " << mSystemSettings.TrackerEdgeDistanceFilter[evalLvl] << " res: " << residual
                           << " resInfo: " << resInfo.nBadEdges << " good: " << resInfo.nGoodEdges;
        return;
    }
    const auto resWeight = getWeightOfEvoR(residual,mSystemSettings.TrackerResidualHuberWeight);
    const auto resSquared = residual*residual;
    resInfo.sumWeightedError += (resWeight * resSquared);
    resInfo.sumError += resSquared;
    ++resInfo.nGoodEdges;
}

std::pair<size_t,size_t>
Tracker::histogramVoting(const FrameData& targetFrame, const FrameData& sourceFrame, const SE3Pose& T_t_s) const
{
    constexpr size_t EvalLvl{0};
    const Mat33f KRK_i = mCameraMatrix.K[EvalLvl]*T_t_s.rotationMatrix().cast<float>()*mCameraMatrix.K_inv[0];
    const Vec3f Kt = mCameraMatrix.K[EvalLvl]*T_t_s.translation().cast<float>();
    size_t nNonHits{0}, nHits{0};
    const Vec3f * const optLvl = targetFrame.getOptStructurePtr(EvalLvl);
    cv::Mat depthImage = targetFrame.mFrameSet->depth;
    const size_t width = mCameraMatrix.width[EvalLvl];
    size_t nEdges{0};
    if (sourceFrame.getEdgesGood().empty())
    {
        //For now, we assume that the detected edges remain untouched
        for (const auto& edgePixel : sourceFrame.getDetectedEdges())
        {
            ++nEdges;
            const Vec2f pt2Dnew = edgePixel->reproject(KRK_i,Kt);
            if (!isInImageGreater(mCameraMatrix.imageBounds[EvalLvl],pt2Dnew))// || new_idepth <= 0)
            {
                ++nNonHits;
                continue;
            }
            const Vec2i pt2DNewInt = pt2Dnew.cast<int>();
            const float depthVal = depthImage.at<float>(pt2DNewInt[1],pt2DNewInt[0]);
            if (isValidDepth(depthVal,mSystemSettings.InputDepthMin,mSystemSettings.InputDepthMax))
            {
                float residual = optLvl[pt2DNewInt[0]+pt2DNewInt[1]*width][0];
                //check distance
                if (residual > mSystemSettings.TrackerHistDist)
                    ++nNonHits;
                else
                    ++nHits;
            }
        }
    }
    else
    {
        cv::Mat edges = sourceFrame.mFrameSet->edgeImage;
        cv::Mat depth = sourceFrame.mFrameSet->depth;
        for (int xx = 0; xx < edges.cols; ++xx)
            for (int yy = 0; yy < edges.rows; ++yy)
            {
                const uint8_t edge = edges.at<uint8_t>(yy,xx);
                if (isEdge(edge))
                {
                    const float depthVal = depth.at<float>(yy,xx);
                    if (isValidDepth(depthVal,mSystemSettings.InputDepthMin,mSystemSettings.InputDepthMax))
                    {
                        const Vec2f pt2Dnew = (KRK_i*Vec3f(xx,yy,1)+Kt*1.0/depthVal).hnormalized();
                        if (!isInImageGreater(mCameraMatrix.imageBounds[EvalLvl],pt2Dnew))// || new_idepth <= 0)
                        {
                            ++nNonHits;
                            continue;
                        }
                        const Vec2i pt2DNewInt = pt2Dnew.cast<int>();
                        const float depthValReproj = depthImage.at<float>(pt2DNewInt[1],pt2DNewInt[0]);
                        if (!isValidDepth(depthValReproj,mSystemSettings.InputDepthMin,mSystemSettings.InputDepthMax))
                        {
                            // ++nNonHits;
                            continue;
                        }
                        const float residual = optLvl[pt2DNewInt[0]+pt2DNewInt[1]*width][0];
                        //check distance
                        if (residual > mSystemSettings.TrackerHistDist)
                            ++nNonHits;
                        else
                            ++nHits;
                    }
                }
            }
    }
    return std::make_pair(nHits,nNonHits);
}


bool
Tracker::histogramVotingAfterTracking(const FrameData& targetFrame, const FrameData& sourceFrame, const SE3Pose& T_t_s, const ResidualInfo& resInfo) const
{
    std::array<size_t,2> overlaps;
    std::fill(overlaps.begin(),overlaps.end(),0);
    constexpr size_t EvalLvl{0};
    const size_t width = mCameraMatrix.width[EvalLvl], 
                 height = mCameraMatrix.height[EvalLvl];
    const Mat33f KRK_i = mCameraMatrix.K[EvalLvl]*T_t_s.rotationMatrix().cast<float>()*mCameraMatrix.K_inv[0];
    const Vec3f Kt = mCameraMatrix.K[EvalLvl]*T_t_s.translation().cast<float>();        
    mCountingMap.setTo(cv::Scalar(0));
    assert(sourceFrame.getEdgesGood().empty() && "Edges not empty!!");
    size_t nValidPixels = 0, nAllPixels = 0;
    //reproject
    for (const auto& edge : sourceFrame.mFrameSet->mValidEdgePixels)
    {
        nAllPixels++;
        const Vec2f pt2Dnew = edge->reproject(KRK_i,Kt);
        if (!isInImageGreater(mCameraMatrix.imageBounds[EvalLvl],pt2Dnew)) continue;
        const Vec2i pt2DNewInt = pt2Dnew.cast<int>();
        mCountingMap.at<uint8_t>(pt2DNewInt[1],pt2DNewInt[0]) = 1;
        nValidPixels++;
    }
    cv::Mat edgeImg = targetFrame.mFrameSet->edgeImage;
    const Vec3f * const optLvl = targetFrame.getOptStructurePtr(EvalLvl);
    cv::Mat depth = targetFrame.mFrameSet->depth;
    cv::Mat debugImg(height,width,CV_8UC1,cv::Scalar(0));

    for (int xx = 0; xx < mCountingMap.cols; ++xx)
        for (int yy = 0; yy < mCountingMap.rows; ++yy)
        {
            const float depthVal = depth.at<float>(yy,xx);
            if (isValidDepth(depthVal))
            {
                const uint8_t mval = mCountingMap.at<uint8_t>(yy,xx);
                // h[mval]++;
                const float dtVal = optLvl[yy*width+xx][0];
                if ((mval == 0 && edgeImg.at<uint8_t>(yy,xx) > 0) || (dtVal < mSystemSettings.TrackerHistDist && mval > 0) )
                    overlaps[mval]++;
                if ((mval == 0 && edgeImg.at<uint8_t>(yy,xx) > 0))//non-hits
                {
                    debugImg.at<uint8_t>(yy,xx) = 128;
                }
                if (dtVal < mSystemSettings.TrackerHistDist && mval > 0)
                {
                    debugImg.at<uint8_t>(yy,xx) = 255;
                }

            }
        }
    auto score = overlaps[1];
    bool isValid = (overlaps[0] < score || (overlaps[0] < score*2 && resInfo.meanErrorGood(true) < 0.5));
    I3D_LOG(i3d::info) << "Quality assessment after tracking: non-hits [0]" << overlaps[0] << " [1]: " << overlaps[1]
                       << " avgError: " << resInfo.meanErrorGood(true) << "score*2: " << score*2 ;
    if (!isValid)
    {
        reprojectToImage(targetFrame,sourceFrame,T_t_s,"reprojHist",false);
        cv::imshow("debugImg",debugImg);
        cv::imshow("mCountingMap",mCountingMap*255);
        cv::imshow("edgeImgTarget", edgeImg);
        cv::imshow("edgeImgSource", sourceFrame.mFrameSet->edgeImage);
        cv::waitKey(0);
    }
    return isValid;
}

bool
Tracker::histogramVotingVerification(const KeyframePoseVector& keyFrames, const FrameData& targetFrame, const Vec4f& histWeights)// const
{
    std::array<size_t,4> overlaps;
    std::fill(overlaps.begin(),overlaps.end(),0);
    constexpr size_t EvalLvl{0};
    const auto width = mCameraMatrix.width[EvalLvl];
    mCountingMapAll.setTo(0);
    // cv::Mat depthVerify = sourceFrame.mFrameSet->depth;
    for (const auto& sourceFramePair : keyFrames)
    {
        const FrameData& sourceFrame = *sourceFramePair.first;
        const SE3Pose T_t_s = sourceFramePair.second.inverse();
        const Mat33f KRK_i = mCameraMatrix.K[EvalLvl]*T_t_s.rotationMatrix().cast<float>()*mCameraMatrix.K_inv[0];
        const Vec3f Kt = mCameraMatrix.K[EvalLvl]*T_t_s.translation().cast<float>();
        mCountingMap.setTo(0);
        const cv::Mat edges = sourceFrame.mFrameSet->edgeImage;
        const cv::Mat depth = sourceFrame.mFrameSet->depth;
        for (int xx = 0; xx < edges.cols; ++xx)
            for (int yy = 0; yy < edges.rows; ++yy)
            {
                const uint8_t edge = edges.at<uint8_t>(yy,xx);
                if (isEdge(edge))
                {
                    const float depthVal = depth.at<float>(yy,xx);
                    if (isValidDepth(depthVal,mSystemSettings.InputDepthMin,mSystemSettings.InputDepthMax))
                    {
                        const Vec2f pt2Dnew = (KRK_i*Vec3f(xx,yy,1)+Kt*1.0/depthVal).hnormalized();
                        if (!isInImageGreater(mCameraMatrix.imageBounds[EvalLvl],pt2Dnew)) continue;
                        const Vec2i pt2DNewInt = pt2Dnew.cast<int>();
                        mCountingMap.at<uint8_t>(pt2DNewInt[1],pt2DNewInt[0]) = 1;
                    }
                }
            }
        mCountingMapAll += mCountingMap;
    }
    const cv::Mat edgeImg = targetFrame.mFrameSet->edgeImage;
    const Vec3f * const optLvl = targetFrame.getOptStructurePtr(EvalLvl);
    for (int xx = 0; xx < mCountingMapAll.cols; ++xx)
        for (int yy = 0; yy < mCountingMapAll.rows; ++yy)
        {
            const uint8_t mval = mCountingMapAll.at<uint8_t>(yy,xx);
            const float dtVal = optLvl[yy*width+xx][0];
            if ((mval == 0 && edgeImg.at<uint8_t>(yy,xx) > 0) || (dtVal < mSystemSettings.TrackerHistDist && mval > 0) )
                overlaps[mval]++;
        }
    
    const auto score = overlaps[1]*histWeights[1]+overlaps[2]*histWeights[2]+overlaps[3]*histWeights[3];
    I3D_LOG(i3d::info) << "o[0]: " << overlaps[0] << " o[1]: " << overlaps[1] << " o[2]: " << overlaps[2]<< " o[3]: " << overlaps[3]
                       << " histWeights: " << histWeights.transpose() << "overlap: " << score;
    I3D_LOG(i3d::info) << "Total: " << targetFrame.mFrameSet->mValidEdgePixels.size();
    return (overlaps[0]*histWeights[0] < score);
}

/**
 * Computes the residuals of each edge
 * In contrast to evaluateCostFunction and calculateErrorsAndBuffers, evaluateCostFunctionFast skips
 *  many intermediate computations and thus is "faster"
 */
float Tracker::evaluateCostFunctionFast(const FrameData& targetFrame, const FrameData& sourceFrame, 
                                        const SE3Pose& T_t_s, ResidualInfo& resInfo) const
{
    resInfo.resetAll();
    constexpr size_t EvalLvl{0};
    //precompute rotations and translations
    const Vec3f Kt = mCameraMatrix.K[EvalLvl]*T_t_s.translation().cast<float>();
    const Mat33f KRK_i = mCameraMatrix.K[EvalLvl]*T_t_s.rotationMatrix().cast<float>()*mCameraMatrix.K_inv[0];
    //alias for intrinsics
    const size_t width = mCameraMatrix.width[EvalLvl], height = mCameraMatrix.height[EvalLvl];
    const Vec3f * const optLvl = targetFrame.getOptStructurePtr(EvalLvl);
    //If the detected edges are untouched, we have not created good edges yet
    if (sourceFrame.getEdgesGood().empty())
    {
        for (const auto& edge : sourceFrame.getDetectedEdges())
        {
            evalReprojection(KRK_i,Kt,edge->getPixel2D(),edge->idepth,optLvl,EvalLvl,resInfo);
        }
    }
    //If we generated good edges, detected edges were already deleted.
    //Thus, we recompute them
    else
    {
        cv::Mat edgeImage = sourceFrame.mFrameSet->edgeImage;
        cv::Mat depthImage = sourceFrame.mFrameSet->depth;
        for (size_t xx = 0; xx < width; ++xx)
            for (size_t yy = 0; yy < height; ++yy)
            {
                const uint8_t edgePixel = edgeImage.at<uint8_t>(yy,xx);
                if (isEdge(edgePixel))
                {
                    const float depthVal = depthImage.at<float>(yy,xx);
                    if (isValidDepth(depthVal,mSystemSettings.InputDepthMin,mSystemSettings.InputDepthMax))
                        evalReprojection(KRK_i,Kt,Vec2f(xx,yy),1.0f/depthVal,optLvl,EvalLvl,resInfo);
                }
            }
    }
    //evaluation is always with float
    return resInfo.meanErrorGood(mSystemSettings.TrackerUseWeightsResidualsForErrorComputation);
}

/**
 * computes the residuals of each edge.
 * if FILL_BUFFERS=true also computes the buffers that are required for Jacobian computation.
 * Note: If you just want to compute the reprojection, use evaluateCostFunctionFast, which skips
 * many intermediate computations.
 */

float Tracker::calculateErrorsAndBuffers(const FrameData& targetFrame, const FrameData&  sourceFrame,
                                         const SE3Pose& T_init, const size_t lvl, ResidualInfo& resInfo, const bool FILL_BUFFERS) const
{
    resInfo.resetAll();
    const auto& validEdgePixels = sourceFrame.getDetectedEdges();
    I3D_LOG(i3d::detail) << "validEdgePixels: " << validEdgePixels.size();
    size_t nResiduals{0};
    const auto resFactor = 2;//( 1 << lvl); //basically takes only every 4th on the lowest pyr lvl ...
    //Edge detections are always on the top level -> inv(K) at lvl 0
    const Mat33f K_inv = mCameraMatrix.K_inv[0];
    const Vec3f t = T_init.translation().cast<float>();
    const Mat33f RK_i = T_init.rotationMatrix().cast<float>()*K_inv;
    const auto fx{mCameraMatrix.fx[lvl]},fy{mCameraMatrix.fy[lvl]},
               cx{mCameraMatrix.cx[lvl]},cy{mCameraMatrix.cy[lvl]};
    const size_t width = mCameraMatrix.width[lvl];
    const Vec3f * const optLvl = targetFrame.getOptStructurePtr(lvl);
    I3D_LOG(i3d::detail) << "reprojBoundariesLvl: " << mCameraMatrix.imageBounds[lvl].transpose();
    for (const auto& edgePixel : validEdgePixels)
    {
        //Skip every Nth residual if the parameter ist set
        if (mSystemSettings.TrackerSkipResdiualsOnLowerLvls)
        {
            nResiduals++;
            if ((nResiduals % resFactor) != 0) continue;
        }
        const Vec2f pt2D = edgePixel->getPixel2D();//Vec2E(edgePixel.hostX,edgePixel.hostY);
        const Vec3f ptHom = pt2D.homogeneous();
        Vec3f newPt3D = RK_i*ptHom + t*edgePixel->idepth;
        const auto z_inv = 1.0f/newPt3D[2];
        const auto new_idepth = edgePixel->idepth/newPt3D[2];
        newPt3D[0] *= z_inv;
        newPt3D[1] *= z_inv;
        const Vec2f newPt2D(fx*newPt3D[0]+cx,fy*newPt3D[1]+cy);
        if (!isInImageGreater(mCameraMatrix.imageBounds[lvl],newPt2D) || new_idepth <= 0)
        {
            ++resInfo.nBadEdges;
            continue;
        }

        //Now, we have to interpolate
        const auto resInterp =  getInterpolatedElement33(optLvl,newPt2D[0],newPt2D[1],width); //residual,dx,dy
        if (std::isnan(resInterp[0]) || std::isnan(resInterp[1]) || std::isnan(resInterp[2]))
        {
            I3D_LOG(i3d::info) << "NAN resInterp!" << resInterp << " " << newPt2D;
            exit(0);
        }
        const float residual = resInterp[0];
        //check distance
        if (residual >= mSystemSettings.TrackerEdgeDistanceFilter[lvl] && mSystemSettings.TrackerUseEdgeFilter)
        {
            ++resInfo.nBadEdges;
            continue;
        }
        const auto resWeight = getWeightOfEvoR(residual,mSystemSettings.TrackerResidualHuberWeight);

        //Compute the derivatives!
        //This might be a bit tricky but we'll see
        const auto resSquared = residual*residual;
        resInfo.sumWeightedError += (resWeight * resSquared);
        resInfo.sumError += resSquared;
        
        if (FILL_BUFFERS)
        {
            buf_warped_x[resInfo.nGoodEdges] = newPt3D[0];
            buf_warped_y[resInfo.nGoodEdges] = newPt3D[1];
            buf_warped_idepth[resInfo.nGoodEdges] = new_idepth;
            buf_warped_dx[resInfo.nGoodEdges] = resInterp[1];//*fx;
            buf_warped_dy[resInfo.nGoodEdges] = resInterp[2];//*fy;
            buf_warped_residual[resInfo.nGoodEdges] = residual;
            buf_warped_weight[resInfo.nGoodEdges] = resWeight;
        }
        ++resInfo.nGoodEdges;
    }
    return resInfo.meanErrorGood(mSystemSettings.TrackerUseWeightsResidualsForErrorComputation);
}

/**
 * Same as compute jacobians but with SSE instructions. Speed-up of around 3 to 4.
 */
void Tracker::computeJacobiansSSE(lsd_slam::LGS6& ls, const ResidualInfo& resInfo, const size_t lvl) const
{
    I3D_LOG(i3d::detail) << "calculateWarpUpdateEdgeSSE" << resInfo.nGoodEdges << " residuals";
    const __m128 fx = _mm_set1_ps(mCameraMatrix.fx[lvl]);
    const __m128 fy = _mm_set1_ps(mCameraMatrix.fy[lvl]);
    ls.initialize();
    size_t idx = 0;
    for(; idx < resInfo.nGoodEdges; idx+=4)
    {
        const __m128 x = _mm_load_ps(buf_warped_x+idx);
        const __m128 y = _mm_load_ps(buf_warped_y+idx);
        // v[0] = iD*fdx + 0;
        // v[1] = 0 + iD*fdy;
        // v[2] = -iD*(x*fdx+y*fdy); = -iD*x*fdx - iD*y*fdy
        // v[3] = -(fdx*x*y)-fdy*(1+y*y) = -fdy-fdy*y*y-fdx*x*y
        // v[4] = fdx * (1 + x*x)+fdy*x*y; = fdx+fdx*x*x + fdy*x*y
        // v[5] = fdy*x-fdx*y;

        // redefine pz
        const __m128 id = _mm_load_ps(buf_warped_idepth+idx); 
        const __m128 dx = _mm_load_ps(buf_warped_dx+idx); //TODO: remove foxal length multi
        const __m128 fxdx = _mm_mul_ps(fx,dx);
        const __m128 J60 = _mm_mul_ps(id,fxdx); //id * dx * fx

        const __m128 dy = _mm_load_ps(buf_warped_dy+idx);
        const __m128 fydy = _mm_mul_ps(fy,dy);
        const __m128 J61 = _mm_mul_ps(id,fydy); //iD*gy *fy
        const __m128 fxdxx = _mm_mul_ps(fxdx,x); //fx*dx*x
        const __m128 fydyy = _mm_mul_ps(fydy,y); //fy*dy*y
        //Note: _mm_xor_ps(vec, _mm_set1_ps(-0.f)) flips the singn
        const __m128 J62 = _mm_mul_ps(_mm_sub_ps(_mm_xor_ps(fxdxx, _mm_set1_ps(-0.f)),fydyy),id);
        const __m128 J63 = _mm_sub_ps(_mm_sub_ps(_mm_xor_ps(fydy, _mm_set1_ps(-0.f)),_mm_mul_ps(fydyy,y)),_mm_mul_ps(fxdxx,y));
        const __m128 J64 = _mm_add_ps(_mm_add_ps(fxdx,_mm_mul_ps(fxdxx,x)),_mm_mul_ps(fydyy,x));
        const __m128 J65 = _mm_sub_ps( _mm_mul_ps(fydy,x), _mm_mul_ps(fxdx,y));
        ls.updateSSE(J60, J61, J62, J63, J64, J65, _mm_load_ps(buf_warped_residual+idx), _mm_load_ps(buf_warped_weight+idx));
    }

    //go through the rest of the points
    for (; idx < resInfo.nGoodEdges; ++idx)
    {
        const float fdx = buf_warped_dx[idx]*mCameraMatrix.fx[lvl];//*fx; //gradient_x * focal_x
        const float fdy = buf_warped_dy[idx]*mCameraMatrix.fy[lvl];//*fy; //gradient_y * focal_y
        const float x = buf_warped_x[idx];
        const float y = buf_warped_y[idx];
        const float iD = buf_warped_idepth[idx]; //1/Z
        lsd_slam::Vector6 v;
        v[0] = iD*fdx;
        v[1] = iD*fdy;
        v[2] = -iD*(x*fdx+y*fdy);
        v[3] = -(fdx*x*y)-fdy*(1+y*y);
        v[4] = fdx * (1 + x*x)+fdy*x*y;
        v[5] = fdy*x-fdx*y;
        ls.update(v, buf_warped_residual[idx], buf_warped_weight[idx]);
    }
    // solve ls
    ls.finish();
}

/**
 * Same as compute Jacobians but with debug outputs
 */
void Tracker::computeJacobiansDebug(lsd_slam::LGS6& ls, const ResidualInfo& resInfo, const size_t lvl) const
{
    I3D_LOG(i3d::detail) << "Computing warp update for "<<resInfo.nGoodEdges<<" residuals"<<ls.error;
    // ls.initialize();
    const auto fx = mCameraMatrix.fx[lvl];
    const auto fy = mCameraMatrix.fy[lvl];
    for (size_t idx = 0; idx < resInfo.nGoodEdges; ++idx)
    {

        const float fdx = buf_warped_dx[idx]*fx;//*fx; //gradient_x * focal_x
        const float fdy = buf_warped_dy[idx]*fy;//*fy; //gradient_y * focal_y
        const float x = buf_warped_x[idx];
        const float y = buf_warped_y[idx];
        const float iD = buf_warped_idepth[idx]; //1/Z
        lsd_slam::Vector6 v;
        v[0] = iD*fdx + 0;
        v[1] = 0 + iD*fdy;
        v[2] = -iD*(x*fdx+y*fdy);
        v[3] = -(fdx*x*y)-fdy*(1+y*y);
        v[4] = fdx * (1 + x*x)+fdy*x*y;
        v[5] = fdy*x-fdx*y;
        if (idx < 20)
        {
            I3D_LOG(i3d::info) << "v: " << v.transpose();
        }
        else return;

    }
}

/**
 * Computes the jacobians with respect to DT, rotation and translation
 * 
 * this is basically J_DT * J_R * J_t
 */
void Tracker::computeJacobians(lsd_slam::LGS6& ls, const ResidualInfo& resInfo, const size_t lvl) const
{
    I3D_LOG(i3d::detail) << "Computing warp update for "<<resInfo.nGoodEdges<<" residuals";
    ls.initialize();
    const auto fx = mCameraMatrix.fx[lvl];
    const auto fy = mCameraMatrix.fy[lvl];
    for (size_t idx = 0; idx < resInfo.nGoodEdges; ++idx)
    {
        const float fdx = buf_warped_dx[idx]*fx;//*fx; //gradient_x * focal_x
        const float fdy = buf_warped_dy[idx]*fy;//*fy; //gradient_y * focal_y
        const float x = buf_warped_x[idx];
        const float y = buf_warped_y[idx];
        const float iD = buf_warped_idepth[idx]; //1/Z
        lsd_slam::Vector6 v;
        v[0] = iD*fdx + 0;
        v[1] = 0 + iD*fdy;
        v[2] = -iD*(x*fdx+y*fdy);
        v[3] = -(fdx*x*y)-fdy*(1+y*y);
        v[4] = fdx * (1 + x*x)+fdy*x*y;
        v[5] = fdy*x-fdx*y;
        ls.update(v, buf_warped_residual[idx], buf_warped_weight[idx]);
    }
    ls.finish();
}

/**
 * Tracks the current frame on a certain lvl by optimizing the reprojection from sourceFrame to targetFrame
 */
Tracker::TrackerStatus Tracker::trackFrameOnLvl(const FrameData& targetFrame, const FrameData& sourceFrame, 
                                                  const size_t lvl, SE3Pose& T_t_s, ResidualInfo& resInfo) const
{
    lsd_slam::LGS6 ls;
    I3D_LOG(i3d::info) << "Computing rel. transformation between: "<< targetFrame.mFrameHeader->mFrameId << "(" 
                       << std::fixed << targetFrame.mFrameSet->mTimestamp << ") and " 
                       << sourceFrame.mFrameHeader->mFrameId << "(" << sourceFrame.mFrameSet->mTimestamp << ") on lvl: " << lvl;
    
    if (sourceFrame.mFrameSet->mValidEdgePixels.size() < 200)
    {
        I3D_LOG(i3d::fatal) << "LOST because pixel < 200";
        return TrackerStatus::Lost;
    }

    ResidualInfo bestRes;
    resInfo.resetAll();
    constexpr bool FillBuffers{true};
    auto lastError = calculateErrorsAndBuffers(targetFrame,sourceFrame,T_t_s,lvl,resInfo,FillBuffers);
    auto error = lastError;
    auto lambda = 0.0f;
    I3D_LOG(i3d::info) << "Transformation before " << lvl << ": " << T_t_s.matrix3x4();
    //now compute iterate until convergence
    for (size_t it{0}; it <  mSystemSettings.TrackerMaxItsPerLvl[lvl]; ++it)
    {
        computeJacobiansSSE(ls,resInfo,lvl);
        int incTry = 0;
        if (resInfo.nGoodEdges < 100)
        {
            I3D_LOG(i3d::fatal) << "LOST because goodEdges < 100 nGoodEdges:" << resInfo.nGoodEdges;
            return TrackerStatus::Lost;
        } 
        while(true) //maybe, this should be removed just to speed up convergence!
        {
            // solve LS system with current lambda
            const Vec6 b = ls.b.cast<double>();
            Mat66 A = ls.A.cast<double>();
            for(size_t i = 0; i < 6; i++) A(i,i) *= 1+lambda;
            const Vec6 inc = A.ldlt().solve(-b);
            incTry++;
            //Make sure that Sophus doesn't crash the program
            if (!inc.allFinite()) return TrackerStatus::Lost;
            SE3Pose T_t_s_new = SE3Pose::exp(inc) * T_t_s;
            error = calculateErrorsAndBuffers(targetFrame,sourceFrame,T_t_s_new,lvl,resInfo,FillBuffers);
            I3D_LOG(i3d::detail) << "inc: " << inc.transpose() << "error: " << error << " lastError: " << lastError;
            if (error < lastError)
            {
                T_t_s = T_t_s_new;
                I3D_LOG(i3d::detail) << "Increment accepted at iteration : "<<it << " lambda: " << lambda << " error: " << error;

                if(error / lastError > mSystemSettings.TrackerConvergenceEps[lvl])
                {
                    I3D_LOG(i3d::info) << "(" << lvl <<", "<< it<< "," << error / lastError <<" ): FINISHED pyramid level " << lvl << " (last residual reduction too small).";
                    it = mSystemSettings.TrackerMaxItsPerLvl[lvl];
                }
                lastError = error;
                lambda = (lambda <= 0.2f ? 0.0f : lambda*0.5f);
                bestRes = resInfo;
                break;
            }
            else
            {
                if(!(inc.dot(inc) > mSystemSettings.TrackerStepSizeMin[lvl]))
                {
                    I3D_LOG(i3d::info) << "(" << lvl <<", "<< it<<"): FINISHED pyramid level " << lvl << " (stepsize too small).";
                    it = mSystemSettings.TrackerMaxItsPerLvl[lvl];
                    break;
                }
                lambda = (lambda < 0.001f ? 0.2f : lambda * (1 << incTry)); // 1 << incTry = 2^incTry
                I3D_LOG(i3d::detail) << "Increment NOT accepted at iteration : "<<it << " lambda: " << lambda;
            }

        }
    }
    resInfo = bestRes;
    I3D_LOG(i3d::detail) << "Transformation after " << lvl << ": " << T_t_s.matrix3x4();
    return TrackerStatus::Ok;
}

/**
 * We analyse the tracking result to determine if we need a new keyframe or have lost tracking
 */
Tracker::TrackerStatus Tracker::analyseTrackingResult(const Tracker::TrackerStatus trackingStatus, const ResidualInfo& resInfo) const
{
    //Something went wrong during tracking
    if (trackingStatus != TrackerStatus::Ok) return TrackerStatus::Lost;
    
    //First, we try to find out if we are lost
    if (resInfo.meanErrorGood(true) > mSystemSettings.TrackerAvgResidualBeforeTrackingLoss)
    {
        I3D_LOG(i3d::info) << "Tracking lost!" << resInfo.meanErrorGood(true) <<" "<<mSystemSettings.TrackerAvgResidualBeforeTrackingLoss;
        return TrackerStatus::Lost;
    }

    //Now, apply some heuristics to check the tracking quality
    const auto opticalFlowThreshold = std::sqrt(resInfo.opticalFlowT/resInfo.nGoodEdges)*mSystemSettings.TrackerOpticalFlowTFactor +
                                      std::sqrt(resInfo.opticalFlowRT/resInfo.nGoodEdges)*mSystemSettings.TrackerOpticalFlowRTFactor;

    I3D_LOG(i3d::info) << "opticalFlowThreshold: " << opticalFlowThreshold << " resInfo good: " << resInfo.nGoodEdges << ", " << resInfo.nBadEdges;
    //Optical flow criteria similar to DSO
    if (opticalFlowThreshold > mSystemSettings.TrackerOpticalFlowThreshold) return TrackerStatus::NewKeyframe;
    //Maybe tracking lost??
    if (resInfo.nGoodEdges < resInfo.nBadEdges*2) return TrackerStatus::NewKeyframe;
    return TrackerStatus::Ok;
}
}
