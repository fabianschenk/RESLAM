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
#include <mutex>
#include <vector>
#include "../config/Defines.h"
#include "../IOWrapper/DataStructures.h"
#include "../GUI/Output3DWrapper.h"

namespace RESLAM {
namespace FernRelocLib { class FernRelocaliser;}
class LoopCloser;
class Relocaliser;
class LocalMapper;
class Tracker;

class Mapper
{
private:

    //Maybe protection not necessary
    mutable std::mutex mAllFrameHeadersMutex;
    std::vector<std::unique_ptr<FrameHeader>> mAllFrameHeaders;
    std::vector<FrameHeader*> mAllKeyFrameHeaders;
    //Maybe protection not necessary
    mutable std::mutex mAllKeyFrameDataMutex;
    std::vector<std::unique_ptr<FrameData>> mAllKeyFrameData;
    std::vector<IOWrap::Output3DWrapper*> mOutputWrapper;
    const SystemSettings& mSystemSettings;
    CameraMatrix mCameraMatrix;

    //Fern relocaliser
    std::unique_ptr<FernRelocLib::FernRelocaliser> mFernRelocaliser;
    std::unique_ptr<LocalMapper> mLocalMapper;
    std::unique_ptr<LoopCloser> mLoopCloser;
    std::unique_ptr<Relocaliser> mRelocaliser;
    std::unique_ptr<Tracker> mLCRelocaliserTracker; ///< Tracker for loop closer and relocaliser
    //TODO: Maybe generate own camera matrix and update it
    ///< The frame id till which we deleted last time. This prevents the removal of the whole map after many tracking losses
    size_t mRelocaliserDeletedTill = 0; 

public:
    bool setToRelocalisationMode(); ///< returns true if a full system reset is necessary!
    explicit Mapper(const SystemSettings& ebSlamSystemSettings);
    ~Mapper();
    bool isLocalMappingInProcess() const;
    void addOutputWrapper(const std::vector<IOWrap::Output3DWrapper*>& outputWrappers);

    const FrameData * getMostRecentKeyframe() const
    {
        std::lock_guard<std::mutex> l(mAllKeyFrameDataMutex);
        return mAllKeyFrameData.back().get();
    }

    void addKeyFrameToAll(std::unique_ptr<FrameData> kf, const bool checkLoop = true);

    auto getNbOfTotalFrames() const 
    {
        std::lock_guard<std::mutex> l(mAllFrameHeadersMutex);
        return mAllFrameHeaders.size();
    }

    auto getNbOfKeyFrames() const 
    {
        std::lock_guard<std::mutex> l(mAllFrameHeadersMutex);
        return mAllKeyFrameHeaders.size();
    }

    void stopMapper();

    KeyframePoseVector getSurroundingFrames(const FrameData& candidateFrame, const SE3Pose& T_cand_curr_init, const size_t nFrames) const;
    
    /**
     * Compute the transformation from frame f1 to frame f2, which is T_f1_f2 = T_f1_W * T_W_f2
     */
    SE3Pose computeRelativeTransformationTf1f2(const FrameData& f1, const FrameData& f2) const
    {
        std::lock_guard<std::mutex> l(mAllFrameHeadersMutex);
        //T_f1_f2 = T_f1_W * T_W_f2
        return f1.mFrameHeader->getWorldToCam()*f2.mFrameHeader->getCamToWorld();
    }

    bool relocaliseFrame(std::unique_ptr<FrameData> newFrameData, std::unique_ptr<FrameHeader> fh, SE3Pose& T_t_s);

    ///Moves a normal frame header to the to mAllFrameHeader
    void addNormalFrameHeaderAndId(std::unique_ptr<FrameHeader> fh);
    ///Adds the header of a keyframe to the mAllKeyFrameHeaders and moves it to mAllFrameHeaders
    void addKeyFrameHeaderAndId(std::unique_ptr<FrameHeader> fh);
    void updatePosesAfterLocalMapper(std::vector<FrameData*>& frameHessians);
    void updatePosesAfterLocalMapperNew(std::vector<FrameData*>& frameHessians);
    FrameData * findLastValidKeyframe(FrameData * startFrame) const;
    FrameData * findLastValidKeyframeNoLock(FrameData * startFrame) const;
    FrameData * findFirstValidKeyframe(FrameData* startFrame) const;
    void changeRefFrame(const FrameData* frame,const FrameData* newRefFrame);
    void changeRefFrameLock(const FrameData* frame,const FrameData* newRefFrame);
    void outputPoses() const;
    PoseVector computePosesToTry() const;
    void updateFramePosesFromNewKeyframePoses(size_t lowestIdInWindow);
    void updateFramePosesFromNewKeyframePosesDebug(size_t lowestIdInWindow);

    size_t getNumberOfFramesInFernDb() const;
    size_t getNumberOfLoopClosures() const;

    PoseVector returnCopyOfKeyFrameWorldPoses();
    void updatePosesAfterLoopClosure(const CeresPoseVector& updatedCamToWorldPoses);
    void copyKeyframePosesAndConstraints(CeresPoseVector& ceresPoseVector, RelPoseConstraints& ceresConstraints);
    void addConstraintsBetweenPoses(RelPoseConstraints& ceresConstraints) const;
    std::vector<std::tuple<double,size_t,size_t>> computeRelDistBetweenKFs() const;
};
}

