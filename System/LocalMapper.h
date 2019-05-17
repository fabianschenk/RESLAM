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

#include "../IOWrapper/DataStructures.h"
#include "../GUI/Output3DWrapper.h"
#include "../Utils/IndexThreadReduce.h"
#include <string>
#include <vector>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <thread>

namespace RESLAM
{
class Mapper;
class WindowedOptimizer;
class CoarseDistanceMap;
struct LMS
{
    static constexpr size_t MinFrameAge{1};
    static constexpr size_t MaxFrames{7};
    static constexpr size_t MinFrames{5};
    // marginalize a frame if less than X% points remain.
    static constexpr float MinPtsRemainingForMarg{0.05}; 
};

class LocalMapper
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        LocalMapper(const SystemSettings& systemSettings);
        ~LocalMapper();
        void reAddConnectionsFromFrames(std::vector<std::unique_ptr<FrameData>>& allKeyFrames);
        void resetLocalMapper(const size_t deleteTillFrameId);
        void localMapping();
        void queueKeyframe(FrameData* newKf);
        void localMappingLinear(FrameData* newKf);
        void stopLocalMapping()
        {
            {
                std::lock_guard<std::mutex> lock(mStopLocalMappingMutex);
                mStopLocalMapping = true;
                I3D_LOG(i3d::info) << "Stopping mStopLocalMapping";
            }
            mQueueCv.notify_one();
        }

        bool hasStoppedLocalMapping() const
        {
            std::lock_guard<std::mutex> lock(mStopLocalMappingMutex);
            return mStopLocalMapping;
        }

        auto isLocalMappingInProgress() const
        {
            std::lock_guard<std::mutex> lock(mIsLocalMappingInProgressMutex);
            return mIsLocalMappingInProgress;
        }
        void setIsLocalMappingInProgress(bool flagIsLocalMappingInProgress)
        {
            std::lock_guard<std::mutex> lock(mIsLocalMappingInProgressMutex);
            mIsLocalMappingInProgress = flagIsLocalMappingInProgress;
        }


        void setOutputWrapper(const std::vector<IOWrap::Output3DWrapper*>& outputWrapper);
        void setParentMapper(Mapper* mapper) { mParentMapper = mapper; }

        /***
         * Fixes the poses of all the keyframes in the window
         */
        void fixWindowPoses(CeresPoseVector& ceresPoseVector) const;

        /**
         * Adds the relative poses of the current window to the ceres constraints
         */
        void addWindowConstraints(RelPoseConstraints& ceresConstraints);

        void printWindowPoses() const;
        void publishGraph();
    private:
        void addFrameConstraintToGraph(const size_t frameId1, const size_t frameId2);
        void addFrameConstraintToGraph(const CeresConstraint& c);
        /// Pointer to mapper, where we update the poses!
        Mapper* mParentMapper; 
        void marginalizeFrame(FrameData* frame);
        /// only exists if LocalMapLinearProcessing == false
        std::thread localMappingThread;
        bool doStepFromBackup(float stepfacC,float stepfacT,float stepfacR,float stepfacA,float stepfacD);
        float optimize(size_t numOptIts);
        double calcLEnergy();
        double calcMEnergy();
        void loadStateBackup();
        void linearizeAll_Reductor(bool fixLinearization, std::vector<EdgeFrameResidual*>* toRemove, int min, int max, Vec10* stats, int tid);
        Vec3 linearizeAll(bool fixLinearization);
        void setPrecalcValues();

        //this is very similar to makeKeyFrame
        void flagFramesForMarginalization(FrameData* newFH);
        void performLocalMapping(FrameData* refFrameNew);
        
        /**
         * Reprojects the good edges from previous keyframes to the newest
         * and adds additional residuals when possible.
         */
        void addNewResidualsForOldPoints(FrameData* refFrameNew);
        void addKeyFrame(FrameData* newKf);
        void activatePointsMT();
        void performWindowedOptimization();
        void computeCurrentMinActDist();
        void activateEdges_Reductor(EdgePixelVec* optimized, DetectedEdgesPtrVec* toOptimize,int min, int max, Vec10* stats, int tid);
        // void activateEdges_Reductor(EdgePixelVec& optimized, const DetectedEdgesPtrVec& toOptimize,int min, int max, Vec10* stats, int tid);
        void applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid);
        //Generates an EdgePixel from a DetectedEdge
        EdgePixel* optimizeImmaturePoint(const DetectedEdge* point, int minObs, ImmatureEdgeTemporaryResidual* residuals);
        void solveSystem(int iteration, double lambda);
        void backupState(bool backupLastStep);
        void removeOutliers();
        void flagPointsForRemoval();
        std::vector<VecX> getNullspaces(std::vector<VecX> &nullspaces_pose, std::vector<VecX> &nullspaces_scale);

        ConnectivityMap mMargConnectivityGraph;
        std::vector<IOWrap::Output3DWrapper*> mOutputWrapper;
        const SystemSettings& mSystemSettings;

        //multi-threading
        bool mStopLocalMapping;
        mutable std::mutex mStopLocalMappingMutex;
        mutable std::mutex mIsLocalMappingInProgressMutex;
        bool mIsLocalMappingInProgress;

        //We might be able to remove one
        mutable std::mutex mFrameHessiansMutex;
        std::vector<FrameData*> mFrameHessians;
        mutable std::mutex mFrameHessiansQueueMutex;
        std::queue<FrameData*> mFrameHessiansQueue;
        std::condition_variable mQueueCv;

        CameraMatrix Hcalib; 
        std::unique_ptr<WindowedOptimizer> mWindowedOptimizer;
        float mCurrentMinActDst = 2;
        std::unique_ptr<CoarseDistanceMap> mCoarseDistanceMap;
        IndexThreadReduce<Vec10> mThreadReduce;
        std::vector<EdgeFrameResidual*> mActiveResiduals;
        bool isLost;
        // statistics
        long int statistics_numForceDroppedResBwd;
        long int statistics_numForceDroppedResFwd;
        float statistics_lastFineTrackRMSE;
};
}
