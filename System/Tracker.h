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
#pragma once

#include <string>
#include "../IOWrapper/DataStructures.h"
#include "../config/Defines.h"
namespace lsd_slam{ class LGS6; }
namespace RESLAM
{
class CameraMatrix;
class SystemSettings;


struct ResidualInfo
{
    size_t nGoodEdges{0};
    size_t nBadEdges{0};
    float sumError{0}; //sum of all errors
    float sumWeightedError{0};
    float opticalFlowT{0}, opticalFlowRT{0};
    void resetAll()
    {
        nGoodEdges = nBadEdges = 0;
        sumError = sumWeightedError = 0;
        opticalFlowRT = opticalFlowT = 0;
    }
    
    //average reprojection error over good and bad edges
    inline auto meanErrorAll(bool useWeighting) const
    {
        if ((nGoodEdges+nBadEdges) == 0) return FLOAT_INF;
        return (useWeighting ? sumWeightedError : sumError)/static_cast<float>(nGoodEdges+nBadEdges);
    }

    //average reprojection error over good edges
    inline auto meanErrorGood(bool useWeighting) const 
    { 
        if (nGoodEdges == 0) return FLOAT_INF;
        return (useWeighting ? sumWeightedError : sumError)/static_cast<float>(nGoodEdges);
    }
};

class Tracker
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        bool flagIterationsDebug = false;
        //The different modes the tracker can return
        enum class TrackerStatus
        {
            Ok, //tracking went smoothly
            Lost, //something went wrong
            NewKeyframe //tracking went smoothly but new keyframe required
        };

        static std::string printTrackerStatus(const TrackerStatus trackerStatus)
        {
            if (trackerStatus == TrackerStatus::Ok) return "Ok";
            if (trackerStatus == TrackerStatus::Lost) return "Lost";
            if (trackerStatus == TrackerStatus::NewKeyframe) return "NewKeyframe";
            return "Unknown TrackerStatus";
        }
        explicit Tracker(const SystemSettings& systemSettings);
        ~Tracker();
        /// finds the initialization according to a motion model and tracks the frames
        /// @params currentFrame: Frame to track
        ///         rest: Transformations from N,N-1,N-2 to the reference frame
        //          return tracking status
        TrackerStatus findInitAndTrackFrames(const FrameData& targetFrame, const FrameData& , SE3Pose& T_ref_N,
                                                const PoseVector &posesToTry) const;
        TrackerStatus trackFrames(const FrameData& targetFrame, const FrameData& sourceFrame, SE3Pose& T_t_s, ResidualInfo& resInfo) const;

        //Reprojects the edges from source to target frame by T_target_source (T_t_s)
        float evaluateCostFunction(const FrameData& targetFrame, const FrameData&  sourceFrame, const SE3Pose& T_t_s) const;
        float evaluateCostFunctionFast(const FrameData& targetFrame, const FrameData& sourceFrame, 
                                           const SE3Pose& T_t_s, ResidualInfo& resInfo) const;

        void evalReprojection(const Mat33f& KRK_i, const Vec3f& Kt, const Vec2f& pt2Dhom, const float idepth, const Vec3f * const optLvl, const size_t evalLvl, ResidualInfo& resInfo) const;
        void setReferenceFrame(FrameData* frameData);
        std::pair<size_t,size_t> histogramVoting(const FrameData& targetFrame, const FrameData& sourceFrame, const SE3Pose& T_t_s) const;
        bool histogramVotingVerification(const KeyframePoseVector& keyFrames, const FrameData& targetFrame, const Vec4f& histWeights);
        bool histogramVotingAfterTracking(const FrameData& targetFrame, const FrameData& sourceFrame, const SE3Pose& T_t_s, const ResidualInfo& resInfo) const;
        void reprojectToImage(const FrameData& targetFrame, const FrameData& sourceFrame, const SE3Pose& T_t_s, const std::string& title = "reproj", const bool WaitFlag = true) const;
    private:
        void computeOpticalFlow(const FrameData& targetFrame, const FrameData& sourceFrame, SE3Pose& T_ref_N, ResidualInfo& resInfo) const;
        float findInit(const FrameData& targetFrame, const FrameData&  sourceFrame, const PoseVector posesToTry, SE3Pose& T_init) const;
        float calculateErrorsAndBuffers(const FrameData& targetFrame, const FrameData&  sourceFrame, const SE3Pose& T_init, const size_t lvl, ResidualInfo& resInfo, const bool FILL_BUFFERS) const;
        void computeJacobians(lsd_slam::LGS6& ls, const ResidualInfo& resInfo, const size_t lvl) const;
        void computeJacobiansDebug(lsd_slam::LGS6& ls, const ResidualInfo& resInfo, const size_t lvl) const;
        void computeJacobiansSSE(lsd_slam::LGS6& ls, const ResidualInfo& resInfo, const size_t lvl) const;
        TrackerStatus trackFrameOnLvl(const FrameData& targetFrame, const FrameData&  sourceFrame, const size_t lvl, SE3Pose& T_ref_N, ResidualInfo& resInfo) const;
        TrackerStatus analyseTrackingResult(const Tracker::TrackerStatus trackingStatus, const ResidualInfo& resInfo) const;
        const SystemSettings& mSystemSettings;
        FrameData* mRefFrame;

        // warped buffers, similar to LSD-SLAM
        //The buffers are currently mutable to allow the methods being const....
        mutable float* buf_warped_idepth;
        mutable float* buf_warped_x;
        mutable float* buf_warped_y;
        mutable float* buf_warped_dx;
        mutable float* buf_warped_dy;
        mutable float* buf_warped_residual;
        mutable float* buf_warped_weight;
        //end warped buffers
        //Will be updated after each LocalMapper optimization
        CameraMatrix mCameraMatrix;
        
        ///< Counting maps for histogram voting
        mutable cv::Mat mCountingMap;
        mutable cv::Mat mCountingMapAll;
};
}
