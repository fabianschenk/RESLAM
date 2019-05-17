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
#include <opencv2/core/mat.hpp>
#include "Mapper.h"
#include "LocalMapper.h"
#include "Tracker.h"
#include "../Utils/timer.h"
#include "../Relocaliser/LoopCloser.h"
#include "../Relocaliser/Relocaliser.h"
#include "../Relocaliser/FernRelocLib/fernrelocaliser.h"
#include "../System/SystemSettings.h"

namespace RESLAM {

bool 
Mapper::isLocalMappingInProcess() const 
{ 
    return (mSystemSettings.EnableLocalMapping ? mLocalMapper->hasStoppedLocalMapping() : false);
}

size_t 
Mapper::getNumberOfFramesInFernDb() const 
{ 
    return (mSystemSettings.EnableLoopClosure ? mLoopCloser->getNumberOfFramesInFernDb() : 0);
}

size_t 
Mapper::getNumberOfLoopClosures() const 
{ 
    return (mSystemSettings.EnableLoopClosure ? mLoopCloser->getNumberOfLoopClosures() : 0);
}

Mapper::Mapper(const SystemSettings& ebSlamSystemSettings):
    mSystemSettings(ebSlamSystemSettings),mCameraMatrix(ebSlamSystemSettings)
{
    if (mSystemSettings.EnableLocalMapping) 
    {
        mLocalMapper = std::make_unique<LocalMapper>(ebSlamSystemSettings);
        mLocalMapper->setParentMapper(this);
    }

    if (mSystemSettings.EnableLoopClosure || mSystemSettings.EnableRelocaliser)
    {
        mLCRelocaliserTracker = std::make_unique<Tracker>(ebSlamSystemSettings);
        mLCRelocaliserTracker->flagIterationsDebug = mSystemSettings.LoopClosureTrackerShowIterations;//false;
        mFernRelocaliser = std::make_unique<FernRelocLib::FernRelocaliser>(cv::Size2i(mCameraMatrix.width[0],mCameraMatrix.height[0]),ebSlamSystemSettings);
    }
    
    if (mSystemSettings.EnableLoopClosure)
        mLoopCloser = std::make_unique<LoopCloser>(ebSlamSystemSettings, mFernRelocaliser.get(),this,mLCRelocaliserTracker.get());
    if (mSystemSettings.EnableRelocaliser)
        mRelocaliser = std::make_unique<Relocaliser>(ebSlamSystemSettings,mFernRelocaliser.get(),this,mLCRelocaliserTracker.get());

}

Mapper::~Mapper() = default;

PoseVector 
Mapper::computePosesToTry() const
{
    PoseVector posesToTry;
    posesToTry.emplace_back(); //Identity
    //Now, we need a lock
    std::unique_lock<std::mutex> lock(mAllFrameHeadersMutex);
    const auto size = mAllFrameHeaders.size();
    //This is currently keyframe to frame
    if (size > 1) //In contrast to DSO, we have not added the newest frame to the list!
    {
        const auto &F_NM1 = mAllFrameHeaders[size-1],
                   &F_NM2 = mAllFrameHeaders[size-2];
        const SE3Pose T_NM1_KF = F_NM1->getWorldToCam()*mAllKeyFrameHeaders.back()->getCamToWorld();
        const SE3Pose T_NM1_NM2 = F_NM1->getWorldToCam()*F_NM2->getCamToWorld();
        lock.unlock();
        posesToTry.push_back(T_NM1_KF); //stop (zero motion from last frame)
        posesToTry.push_back(T_NM1_NM2*T_NM1_KF); //constant motion
        posesToTry.push_back(SE3Pose::exp(T_NM1_NM2.log()*0.5)*T_NM1_KF);//half motion
        posesToTry.push_back(T_NM1_NM2*T_NM1_NM2*T_NM1_KF);//double motion (e.g. frame skip)
    }

    if (mSystemSettings.TrackerTrackFromFrameToKf)
    {
        for (auto& pose : posesToTry)
            pose = pose.inverse();
    }
    return posesToTry;
}

KeyframePoseVector 
Mapper::getSurroundingFrames(const FrameData& candidateFrame, const SE3Pose& T_cand_curr_init, const size_t nFrames) const
{
    if (nFrames % 2 == 0) 
    {    I3D_LOG(i3d::info) << "nFrames should be odd!";}
    
    const size_t start = [&](){
                            if (candidateFrame.mKeyFrameId == 0) return  static_cast<size_t>(0);
                            const size_t nOffset = std::floor(nFrames/2.0);
                            //e.g. we want to find the 5 surrounding frames, we have to go 2 left and 2 right
                            return (candidateFrame.mKeyFrameId < nOffset ? static_cast<size_t>(0) : candidateFrame.mKeyFrameId-nOffset);
                        }();
    
    //Fixme: check if end is greater than the size of the vector!!
    const size_t end = start + nFrames;
    I3D_LOG(i3d::info) << "Surrounding frames start: " << start << " end " << end;

    KeyframePoseVector kfPoseVec;
    //FIXME: Rethink the locks
    // don't actually take the locks yet

    std::unique_lock<std::mutex> l1(mAllFrameHeadersMutex, std::defer_lock), l2(mAllKeyFrameDataMutex, std::defer_lock);
    I3D_LOG(i3d::info) << "lock surrounding frames";
    std::lock(l1,l2); //lock both at the same time
    if (end > mAllKeyFrameData.size())
    {
        I3D_LOG(i3d::error) << "End is wrong!";
        exit(0);
    }
    const SE3Pose T_W_cand = candidateFrame.mFrameHeader->getCamToWorld();
    //Compute the relative transformations T_kf_cand = T_kf_W * T_W_cand
    for (auto idx = start; idx < end; ++idx)
    {
        FrameData* kf = mAllKeyFrameData[idx].get();
        kfPoseVec.emplace_back(kf,kf->mFrameHeader->getWorldToCam()*T_W_cand * T_cand_curr_init);
    }
   
    return kfPoseVec;
}

bool
Mapper::relocaliseFrame(std::unique_ptr<FrameData> frameData, std::unique_ptr<FrameHeader> fh, SE3Pose& T_t_s)
{
    I3D_LOG(i3d::info) << " Trying to relocalise: " << std::fixed << frameData->mFrameSet->mTimestamp;
    auto id = mRelocaliser->relocalise(frameData.get());
    if (id < 0) return false;
    FrameData* candidateFrame = mFernRelocaliser->convertDbId2KeyFrameId(id);
    I3D_LOG(i3d::info) << "Found similar frame in relocalisation mode with id: " << id;
    ResidualInfo resInfo;
    //Track from current to candidate frame
    const auto trackerStatus = mLCRelocaliserTracker->trackFrames(*candidateFrame,*frameData,T_t_s,resInfo);
    if (trackerStatus == Tracker::TrackerStatus::Ok)
    {
        fh->setCamToRef(T_t_s);
        fh->setRefFrame(candidateFrame);
        I3D_LOG(i3d::info) << "Tracking Status: " << Tracker::printTrackerStatus(trackerStatus) << " T_t_s: " << T_t_s.matrix3x4();
        addKeyFrameHeaderAndId(std::move(fh));
        addKeyFrameToAll(std::move(frameData),false);
    }
    return (trackerStatus == Tracker::TrackerStatus::Ok);
}

bool
Mapper::setToRelocalisationMode()
{
    //Evaluate if this makes any difference 
    std::lock_guard<std::mutex> l(mAllFrameHeadersMutex);
    
    //We remove the past "N" frames just to make sure that we do not incorporate any "wrong" frames
    if (mSystemSettings.RelocaliserMinNumberOfKeyframes < mAllKeyFrameData.size())
    {
        I3D_LOG(i3d::info) << "Deleting only a few kfs!";
        //Remove N past frames
        const size_t deleteTillIdx = mAllFrameHeaders.size() - mSystemSettings.RemoveNFramesAfterTrackLoss;
        //First delete the keyframes
        size_t deleteTillFrameId = mAllFrameHeaders[deleteTillIdx]->mFrameId;
        //prevent the deletion of too many frames
        if (deleteTillFrameId < mRelocaliserDeletedTill)
            deleteTillFrameId = mRelocaliserDeletedTill;
        else
            mRelocaliserDeletedTill = deleteTillFrameId;

        size_t delId;
        
        {
            I3D_LOG(i3d::info) << "lock relocalisation mode!";
            std::lock_guard<std::mutex> l(mAllKeyFrameDataMutex);
            mLocalMapper->resetLocalMapper(deleteTillFrameId);
            auto resIt = std::find_if(mAllKeyFrameData.rbegin(),mAllKeyFrameData.rend(),[id = deleteTillFrameId] (const std::unique_ptr<FrameData>& kfData) {return kfData->mFrameHeader->mFrameId < id;});//,
            for (const auto& kf : mAllKeyFrameData)
            {
                I3D_LOG(i3d::info) << "kf before delete: " << kf->mKeyFrameId << " fId: " << kf->mFrameHeader->mFrameId << " tillId: " << deleteTillFrameId;
            }
            delId = std::distance(mAllKeyFrameData.begin(),resIt.base());
            mAllKeyFrameData.erase(resIt.base(),mAllKeyFrameData.end());
            for (const auto& kf : mAllKeyFrameData)
            {
                I3D_LOG(i3d::info) << "kf after delete: " << kf->mKeyFrameId << " fId: " << kf->mFrameHeader->mFrameId << " tillId: " << deleteTillFrameId;
            }

            //Remove all the connections with id >= delete tillFrameid
            for (auto& kf : mAllKeyFrameData)
            {
                I3D_LOG(i3d::info) << "kf: " << kf->mKeyFrameId << " size: " << mAllKeyFrameData.size() << "\n"
                                   << "lastKf: " << mAllKeyFrameData.back()->mKeyFrameId;
                kf->removeConstraintsTill(mAllKeyFrameData.back()->mKeyFrameId);
                for (const auto& c : kf->getConstraints())
                {
                    I3D_LOG(i3d::info) << "c: " << c.i << "/" << c.j;
                }
            }
            //Read the connections
            mLocalMapper->reAddConnectionsFromFrames(mAllKeyFrameData);
            mAllKeyFrameHeaders.erase(mAllKeyFrameHeaders.begin()+delId,mAllKeyFrameHeaders.end());
            //and then all the headers
            mAllFrameHeaders.erase(std::end(mAllFrameHeaders)- mSystemSettings.RemoveNFramesAfterTrackLoss,
                                std::end(mAllFrameHeaders));
            for (auto& ow : mOutputWrapper)
                ow->deleteTillFrameId(deleteTillFrameId);
            mFernRelocaliser->resetFernRelocaliser(mAllKeyFrameData);
        }
        mLoopCloser->resetLoopCloser(mAllKeyFrameData.back()->mKeyFrameId);
        //now, republish the graphs
        mLoopCloser->publishGraph();
        mLocalMapper->publishGraph();
    }
    else
    {
        I3D_LOG(i3d::info) << "Restart all!";
        //we restart all over again
        {

            constexpr size_t deleteTillFrameId{0}; //all
            mLocalMapper->resetLocalMapper(deleteTillFrameId); 
            I3D_LOG(i3d::info) << "lock reset!";
            std::lock_guard<std::mutex> l(mAllKeyFrameDataMutex);
            mAllKeyFrameData.clear();
            mAllKeyFrameHeaders.clear();
            mAllFrameHeaders.clear();
            for (auto& ow : mOutputWrapper)
                ow->deleteTillFrameId(deleteTillFrameId);
            mFernRelocaliser->resetFernRelocaliser(mAllKeyFrameData);
        }
        mLoopCloser.reset(new LoopCloser(mSystemSettings,mFernRelocaliser.get(),this,mLCRelocaliserTracker.get()));
        return true;    
    }
    return false;
}

void
Mapper::addNormalFrameHeaderAndId(std::unique_ptr<FrameHeader> fh)
{
    if (fh != nullptr)
    {
        std::lock_guard<std::mutex> lock(mAllFrameHeadersMutex);
        fh->mFrameId = mAllFrameHeaders.size();
        //We assume that the first frame is always a kf, thus mAllKeyFrameHeaders is never empty
        const SE3Pose camToWorld = mAllKeyFrameHeaders.back()->getCamToWorld()*fh->getCamToRef();
        fh->updateCamToWorld(camToWorld);
        //Publish to all the outputs
        for (auto& wrapper : mOutputWrapper)
        {
            wrapper->publishCamPose(*fh, mCameraMatrix);
        }
        mAllFrameHeaders.push_back(std::move(fh));
    }
}

std::vector<std::tuple<double,size_t,size_t>>
Mapper::computeRelDistBetweenKFs() const
{
    std::vector<std::tuple<double,size_t,size_t>> dist;
    size_t lastIdx = 0;
    I3D_LOG(i3d::info) << "lock compute distance!";
    std::lock_guard<std::mutex> l(mAllKeyFrameDataMutex);
    for (size_t idx = 1; idx < mAllKeyFrameData.size(); ++idx)
    {
        const auto& frame = mAllKeyFrameData[idx];
        if (frame->isInFernDbOrLoop())
        {
            const auto& lastFrame = mAllKeyFrameData[lastIdx];
            const SE3Pose T_last_curr = lastFrame->getPRE_worldToCam()*frame->getPRE_camToWorld();
        
            dist.emplace_back(std::make_tuple(T_last_curr.translation().norm(),lastIdx,idx));
            lastIdx = idx;
        }
    }
    return dist;
}

/**
 * Copies the relative transformation constraints
 */
void Mapper::copyKeyframePosesAndConstraints(CeresPoseVector& ceresPoseVector, RelPoseConstraints& ceresConstraints)
{
    {   
        std::lock_guard<std::mutex> l(mAllFrameHeadersMutex);
        std::transform(mAllKeyFrameHeaders.begin(),mAllKeyFrameHeaders.end(),std::back_inserter(ceresPoseVector),[](const auto& fh){ return fh->getCamToWorld(); }); 
    }
    {
        I3D_LOG(i3d::info) << "lock copy keyframes";
        {
            std::lock_guard<std::mutex> l(mAllKeyFrameDataMutex);
            //Copy constraints from each KF to the ceres constraints
            for (const auto& kf : mAllKeyFrameData)
            {
                const auto& constraints = kf->getConstraints();
                std::copy(constraints.begin(),constraints.end(),std::back_inserter(ceresConstraints));            
            }
        }
        I3D_LOG(i3d::info) << "unlock copy keyframes";

        if (mSystemSettings.EnableLocalMapping)
        {
            if (mSystemSettings.LoopClosureFixWindowPoses)
            {            
                mLocalMapper->fixWindowPoses(ceresPoseVector);
                mLocalMapper->addWindowConstraints(ceresConstraints);
            }
            else
                mLocalMapper->addWindowConstraints(ceresConstraints);
            if (mSystemSettings.LoopClosureAddAdjacentConstraints)
                addConstraintsBetweenPoses(ceresConstraints);
        }
        else
        {      
            addConstraintsBetweenPoses(ceresConstraints);
        }
    }
}

void
Mapper::addConstraintsBetweenPoses(RelPoseConstraints& ceresConstraints) const
{
    //this should never happen
    if (mAllKeyFrameData.empty()) return;
    for (size_t frameId = 0; frameId < mAllKeyFrameData.size()-1; ++frameId)
    {
        const auto& kfBegin = mAllKeyFrameData[frameId];
        const auto& kfEnd = mAllKeyFrameData[frameId+1];
        const SE3Pose T_eb = kfEnd->mFrameHeader->getWorldToCam()*kfBegin->mFrameHeader->getCamToWorld();
        ceresConstraints.emplace_back(T_eb,kfBegin->mKeyFrameId,kfEnd->mKeyFrameId);
    }
}

void 
Mapper::updatePosesAfterLoopClosure(const CeresPoseVector& updatedCamToWorldPoses)
{
    if (mSystemSettings.EnableLocalMapping) mLocalMapper->printWindowPoses();
    std::lock_guard<std::mutex> l(mAllFrameHeadersMutex);
    {
        I3D_LOG(i3d::info) << "lock updatePosesAfterLoopClosure frames mode!";
        std::lock_guard<std::mutex> l1(mAllKeyFrameDataMutex);
        I3D_LOG(i3d::detail) << "Window poses before: " << updatedCamToWorldPoses.size() << "/"<< mAllKeyFrameData.size();
        for (size_t idx = 0; idx < updatedCamToWorldPoses.size(); ++idx)
        {
            auto& kf = mAllKeyFrameData[idx];
            I3D_LOG(i3d::detail) << "id: " << idx << "/"<<kf->mKeyFrameId << " prepose orig: " << kf->getPRE_camToWorld().matrix3x4() 
                               << " pose orig: " << kf->mFrameHeader->getCamToWorld().matrix3x4()
                               << " pose upd: " << updatedCamToWorldPoses[idx].returnPose().matrix3x4()
                               << " kf: " << kf.get() << " " << kf->isInFernDb() << "/" << kf->isInLoop()
                               << " frameId: " << kf->mFrameHeader->mFrameId;
            kf->mFrameHeader->updateCamToWorld(updatedCamToWorldPoses[idx].returnPose());
            kf->setPRE_camToWorld(updatedCamToWorldPoses[idx].returnPose());
            if (!mSystemSettings.LoopClosureFixWindowPoses)
                kf->setEvalPT_scaled(kf->mFrameHeader->getWorldToCam());
        }
    }
    
    
    I3D_LOG(i3d::detail) << "Updating: " << mAllFrameHeaders.size() << " with " << updatedCamToWorldPoses.size() << "poses!";
    
    //Now update all the poses with modified world poses of the keyframes
    updateFramePosesFromNewKeyframePoses(mAllFrameHeaders.front()->mFrameId);
    for (const auto& ow :  mOutputWrapper)
        ow->updateKeyframePoses(mAllKeyFrameHeaders);
    I3D_LOG(i3d::detail) << "Window poses before: ";
    if (mSystemSettings.EnableLocalMapping) mLocalMapper->printWindowPoses();
}

void 
Mapper::addKeyFrameHeaderAndId(std::unique_ptr<FrameHeader> fh)
{
    I3D_LOG(i3d::info) << "addKeyFrameHeaderAndId" << std::fixed << " timestamp: " << fh->mTimestamp;
    if (fh != nullptr)
    {
        std::lock_guard<std::mutex> lock(mAllFrameHeadersMutex);
        fh->mFrameId = mAllFrameHeaders.size();
        //If not empty, compute the world pose
        if (!mAllKeyFrameHeaders.empty()) fh->updateCamToWorld(mAllKeyFrameHeaders.back()->getCamToWorld()*fh->getCamToRef());
        mAllKeyFrameHeaders.push_back(fh.get());
        mAllFrameHeaders.push_back(std::move(fh));
    }
}

void
Mapper::outputPoses() const
{
    if (!mSystemSettings.OutputPoses) return;
    std::ofstream poseFile, poseFileKf;
    //TODO: Think about how to name the pose files
    const std::string poseFileName = (mSystemSettings.OutputPoseFileFolder+"poses.txt");
    const std::string poseFileNameKf = (mSystemSettings.OutputPoseFileFolder+"poses_kf.txt");
    poseFile.open(poseFileName.c_str(),std::ios_base::out);
    poseFileKf.open(poseFileNameKf.c_str(),std::ios_base::out);
    I3D_LOG(i3d::info) << "Trying to open: " << poseFileName;
    if (poseFile.is_open())
    {
        {
            std::lock_guard<std::mutex> lock(mAllFrameHeadersMutex);
            for (const auto& frameHeader : mAllFrameHeaders)
            {
                const auto params = frameHeader->getCamToWorld().params();
                poseFile << std::fixed << frameHeader->mTimestamp << " " << std::setprecision(9) << params[4] << " " << params[5] << " " << params[6] << " " //translation
                         <<  params[0] << " " << params[1] << " " << params[2] << " " <<params[3] << std::endl;
            }
        }
        poseFile.close();
    }
    else
    {
        I3D_LOG(i3d::info) << "Couldn't open pose file: " << poseFileName;
    }

    if (poseFileKf.is_open())
    {
        {
            std::lock_guard<std::mutex> lock(mAllFrameHeadersMutex);
            for (const auto& frameHeader : mAllKeyFrameHeaders)
            {
                const auto params = frameHeader->getCamToWorld().params();
                poseFileKf << std::fixed << frameHeader->mTimestamp << " " << std::setprecision(9) << params[4] << " " << params[5] << " " << params[6] << " " //translation
                           <<  params[0] << " " << params[1] << " " << params[2] << " " <<params[3] << std::endl;
            }
        }
        poseFileKf.close();
    }
}

void 
Mapper::addKeyFrameToAll(std::unique_ptr<FrameData> kf, const bool checkLoop)
{
    kf->makeKeyFrame(mCameraMatrix);
    I3D_LOG(i3d::info) << "addKeyFrameHeaderAndId" << kf->mKeyFrameId << std::fixed << " timestamp: " << kf->mFrameHeader->mTimestamp;
    FrameData* kfPtr = kf.get();
    if (kf != nullptr)
    {
        {
            I3D_LOG(i3d::info) << "lock addKeyFrameToAll frames mode!";
            std::lock_guard<std::mutex> lock(mAllKeyFrameDataMutex);
            kf->mKeyFrameId = mAllKeyFrameData.size();
            kf->setEvalPT_scaled(kf->mFrameHeader->getWorldToCam());
            mAllKeyFrameData.push_back(std::move(kf));
        }
        if (!mSystemSettings.EnableLocalMapping)
        {
            for (auto& wrapper : mOutputWrapper)
            {
                I3D_LOG(i3d::info) << "kfPtr: " << kfPtr;
                wrapper->publishNewKeyframe(kfPtr,mCameraMatrix);
            }
        }
        else
        {
            const auto startTime = Timer::getTime();
            if (mSystemSettings.LocalMapperLinearProcessing)
                mLocalMapper->localMappingLinear(kfPtr);
            else
                mLocalMapper->queueKeyframe(kfPtr);
            const auto endTime = Timer::getTime();
             I3D_LOG(i3d::info) << "Time for local mapping: " << Timer::getTimeDiffMiS(startTime,endTime);
        }
        if (mSystemSettings.EnableLoopClosure && checkLoop)
        {
            I3D_LOG(i3d::info) << "loopCloserLinear" << kfPtr;
            const auto startTime = Timer::getTime();
            mLoopCloser->loopCloserLinear(kfPtr);
            const auto endTime = Timer::getTime();
            I3D_LOG(i3d::info) << "Time for loop closer: " << Timer::getTimeDiffMiS(startTime,endTime);
        }

        {
            std::lock_guard<std::mutex> l(mAllFrameHeadersMutex);
            mOutputWrapper[0]->publishCompleteTrajectory(mAllFrameHeaders);
        }
        I3D_LOG(i3d::info) << "STATUS: frames: " << mAllFrameHeaders.size() << " keyframes " << mAllKeyFrameHeaders.size();
    }
}


void
Mapper::updateFramePosesFromNewKeyframePoses(size_t lowestIdInWindow)
{
    //Assume that first frame is a keyframe!
    for (size_t idx = lowestIdInWindow; idx < mAllFrameHeaders.size(); ++idx)
    {
        auto& frame = mAllFrameHeaders[idx];
        //This only updates non-keyframes
        if (!frame->mIsKeyFrame) 
        {
            I3D_LOG(i3d::detail) << "keyframe" << frame->getRefFrame()->mFrameHeader->mFrameId
                    << " frame: " << frame->mFrameId << " pose: " << frame->getCamToWorld().matrix3x4()
                    << " upd: " << (frame->getRefFrame()->mFrameHeader->getCamToWorld()*frame->getCamToRef()).matrix3x4()
                    << " camToRef: " << frame->getCamToRef().matrix3x4();
            frame->updateCamToWorld(frame->getWorldPoseFromRef());
            I3D_LOG(i3d::detail) << "Updating: " << frame->mFrameId << " from: " << frame->getRefFrame()->mFrameHeader->mFrameId;
            I3D_LOG(i3d::detail) << "after world Update: " << frame->getCamToWorld().matrix3x4();
        }
        //what happens to keyframes that are not in frame hessian?       
    }
}

//TODO: Delete
void
Mapper::updateFramePosesFromNewKeyframePosesDebug(size_t lowestIdInWindow)
{
    LOG_THRESHOLD(i3d::nothing);
    // auto* keyFrameHeader = frameHessians.front()->mFrameHeader;
    // for (size_t idx = keyFrameHeader->mFrameId+1; idx < mAllFrameHeaders.size(); ++idx)#

    //Assume that first frame is a keyframe!
    // auto* keyFrameHeader = mAllFrameHeaders.front().get();
    
    // for (auto& frame : mAllFrameHeaders)
    for (size_t idx = lowestIdInWindow; idx < mAllFrameHeaders.size(); ++idx)
    {
        auto& frame = mAllFrameHeaders[idx];
        if (!frame->mIsKeyFrame) 
        {
            I3D_LOG(i3d::info) << " frame: " << frame->mFrameId
                                << "keyframe" << frame->getRefFrame()->mFrameHeader->mFrameId
                                << " pose: " << frame->getCamToWorld().matrix3x4()
                                << " pose kf: " << frame->getRefFrame()->mFrameHeader->getCamToWorld().matrix3x4()
                                << " upd: " << (frame->getRefFrame()->mFrameHeader->getCamToWorld()*frame->getCamToRef()).matrix3x4()
                                << " camToRef: " << frame->getCamToRef().matrix3x4();
            frame->updateCamToWorld(frame->getWorldPoseFromRef());
            I3D_LOG(i3d::info) << "Updating: " << frame->mFrameId << " from: " << frame->getRefFrame()->mFrameHeader->mFrameId
                               << "after world Update: " << frame->getCamToWorld().matrix3x4();
            I3D_LOG(i3d::info) << "frame: " << frame->mFrameId << " is not a keyframe" << frame->getRefFrame()->mKeyFrameId
                               << " timestamp: " << std::fixed << frame->mTimestamp 
                               << " kfTimestamp: " << std::fixed << frame->getRefFrame()->mFrameHeader->mTimestamp;
        }
        else
        {
            I3D_LOG(i3d::info) << "frame: " << frame->mFrameId << " seems to be a keyframe" << frame->getRefFrame()->mKeyFrameId
                               << " timestamp: " << std::fixed << frame->mTimestamp 
                               << " kfTimestamp: " << std::fixed << frame->getRefFrame()->mFrameHeader->mTimestamp;
        }
    }
    LOG_THRESHOLD(i3d::info);
}

void
Mapper::changeRefFrameLock(const FrameData* frame, const FrameData* newRefFrame)
{
    std::lock_guard<std::mutex> lock(mAllFrameHeadersMutex);
    changeRefFrame(frame, newRefFrame);
}

void
Mapper::changeRefFrame(const FrameData* frame, const FrameData* newRefFrame)
{
    const auto& fh = frame->mFrameHeader;
    for (auto idx = fh->mFrameId+1; idx < mAllFrameHeaders.size();++idx)
    {
        auto& fhCurr = mAllFrameHeaders[idx];
        if (fhCurr->getRefFrame() != frame) break;
        I3D_LOG(i3d::info) << "Updating: " << fhCurr->mFrameId << " from old kf id: " << frame->mKeyFrameId << "/" << frame->mFrameHeader->mFrameId 
                           << " with new ref frame: " << newRefFrame->mFrameHeader->mFrameId << "/" << newRefFrame->mKeyFrameId;
        fhCurr->setCamToRef(fh->getCamToRef()*fhCurr->getCamToRef());
        fhCurr->setRefFrame(newRefFrame);
        fhCurr->updateCamToWorld(newRefFrame->getPRE_camToWorld()*fhCurr->getCamToRef());
    }
}

FrameData * 
Mapper::findLastValidKeyframeNoLock(FrameData * startFrame) const
{
    if (startFrame->mKeyFrameId == 0) return startFrame;
    //We search for the closest frame to the left
    for (int idx = static_cast<int>(startFrame->mKeyFrameId-1); idx >= 0; idx--)
    {
        FrameData * f = mAllKeyFrameData[idx].get();
        I3D_LOG(i3d::info) << "f: " << f->mKeyFrameId << "/"<<f->mFrameHeader->mFrameId <<": " << f->isInFernDb() << "/" <<f->isInLoop();
        if (f->isInFernDbOrLoop()) { return f; } 
    }
    return startFrame;
}

FrameData * 
Mapper::findLastValidKeyframe(FrameData * startFrame) const
{
    I3D_LOG(i3d::info) << "lock findLastValidKeyframe frames mode!";
    std::unique_lock<std::mutex> lock(mAllKeyFrameDataMutex);
    return findLastValidKeyframeNoLock(startFrame);
}
/**
 * 
 * Search through the keyframes till the first kf that is also in the fern db
 * 
 */ 

FrameData*
Mapper::findFirstValidKeyframe(FrameData* startFrame) const
{
    I3D_LOG(i3d::info) << "findFirstValidKeyframe lock!" << startFrame << " " << startFrame->isInFernDb() << "/" << startFrame->isInLoop();
    std::lock_guard<std::mutex> lock(mAllKeyFrameDataMutex);
    FrameData* frameLeft = nullptr;
    //Search to the right
    for (int idx = static_cast<int>(startFrame->mKeyFrameId-1); idx >= 0; --idx)
    {
        auto& frame = mAllKeyFrameData[idx];
        if (frame->isInFernDbOrLoop())
        {
            frameLeft = frame.get();
            break;
        }
    }
    if (frameLeft != nullptr)
        {I3D_LOG(i3d::info) << "Found frameLeft: " << frameLeft->mKeyFrameId;}
    FrameData* frameRight = nullptr;

    //Search to the left
    for (size_t idx = static_cast<int>(startFrame->mKeyFrameId+1); idx < mAllKeyFrameData.size(); ++idx)
    {
        auto& frame = mAllKeyFrameData[idx];
        if (frame->isInFernDbOrLoop())
        {
            frameRight = frame.get();
            break;
        }
    }
    
    I3D_LOG(i3d::info) << "Found frameRight: " << frameRight << " mAllKeyFrameData: " << mAllKeyFrameData.size();
    if (frameLeft == nullptr && frameRight == nullptr)
    {
        I3D_LOG(i3d::info) << "Not valid frame found!";
        exit(0);
        return nullptr;
    } 
    const auto diffLeft = frameLeft == nullptr ? 10000 : (startFrame->mKeyFrameId - frameLeft->mKeyFrameId);
    const auto diffRight = frameRight == nullptr ? 10000 : (frameRight->mKeyFrameId - startFrame->mKeyFrameId);
    I3D_LOG(i3d::info) << "diffLeft: " << diffLeft << " diffRight: " << diffRight;
    return (diffLeft < diffRight ? frameLeft : frameRight);    
}

void
Mapper::updatePosesAfterLocalMapperNew(std::vector<FrameData*>& frameHessians)
{
    I3D_LOG(i3d::info) << "Waiting for lock! updatePosesAfterLocalMapper";
    std::lock_guard<std::mutex> lock(mAllKeyFrameDataMutex);

    //TODO:!!!
    //Now compute the relative poses T_1'2',...,T_1'N
    //Set T_W1' to T_W1

    //This update is also not correct!
    //We have to set the first pose to its old value but keep the relative transformations

    const auto& T_W1  = frameHessians.front()->mFrameHeader->getCamToWorld(); //T_W1 -> previous world pose
    const auto& T_W1n  = frameHessians.front()->mFrameHeader->getCamToWorld(); //T_W1' -> updated world pose
    const auto T_1nW = T_W1n.inverse();
    for (auto* frame : frameHessians)
    {
        const auto newPose = T_W1*T_1nW*frame->getPRE_camToWorld();
        frame->mFrameHeader->updateCamToWorld(newPose);
        frame->setPRE_camToWorld(newPose);
        if (!mSystemSettings.LoopClosureFixWindowPoses)
            frame->setEvalPT_scaled(newPose.inverse());
    }

    //Here, we have a problem since we do not update keyframes that are in the db but not in the window
    for (size_t idx = frameHessians.front()->mKeyFrameId+1; idx < mAllKeyFrameData.size(); ++idx)
    {
        auto& frameData = mAllKeyFrameData[idx];
        auto& frame = frameData->mFrameHeader;
        
        if (frame->mIsKeyFrame)
        {
            I3D_LOG(i3d::info) << "Not updating: " << frame->mFrameId << " refId: " << frame->getRefFrame()->mFrameHeader->mFrameId
                                << " with ref frame: " << frame->getRefFrame()->mKeyFrameId << " marg at: " << frame->marginalizedAt;
            const auto& constraints  = frame->getRefFrame()->getConstraints();
            for (const auto& c : constraints)
            {
                I3D_LOG(i3d::info) << c.i << " " << c.j << "self: " << frame->mFrameId << "/" << frameData->mKeyFrameId
                                   << " marg: " << frame->marginalizedAt;
                const auto& res = std::find_if(frameHessians.cbegin(),frameHessians.cend(),[&](const auto& fh){return fh->mKeyFrameId == c.j;});
                if (res != frameHessians.cend())
                {
                    I3D_LOG(i3d::info) << "Found constraint!" << (*res)->mKeyFrameId; 
                    //Now update this keyframe
                    frame->updateCamToWorld(mAllKeyFrameHeaders[c.j]->getCamToWorld()*c.T_j_i);
                }
            }
        }
    }


    //Now, update all the frames in between
    updateFramePosesFromNewKeyframePoses(frameHessians.front()->mFrameHeader->mFrameId);
    //Send it to the output
    for (auto& ow : mOutputWrapper)
        ow->updateKeyframePoses(mAllKeyFrameHeaders);
    // mOutputWrapper[0]->updateKeyframePoses(mAllKeyFrameHeaders);
}

void
Mapper::updatePosesAfterLocalMapper(std::vector<FrameData*>& frameHessians)
{
    I3D_LOG(i3d::info) << "Waiting for lock! updatePosesAfterLocalMapper";
    std::lock_guard<std::mutex> lock(mAllKeyFrameDataMutex);

    //TODO:!!!
    //Now compute the relative poses T_1'2',...,T_1'N
    //Set T_W1' to T_W1

    //This update is also not correct!
    for (auto* frame : frameHessians)
    {
        frame->mFrameHeader->updateCamToWorld(frame->getPRE_camToWorld());
    }
    //Now, update all the frames in between
    updateFramePosesFromNewKeyframePoses(frameHessians.front()->mFrameHeader->mFrameId);
    //Send it to the output
    for (auto& ow : mOutputWrapper)
        ow->updateKeyframePoses(mAllKeyFrameHeaders);
}
void 
Mapper::addOutputWrapper(const std::vector<IOWrap::Output3DWrapper*>& outputWrappers)
{
    std::copy(outputWrappers.begin(),outputWrappers.end(),std::back_inserter(mOutputWrapper));
    if (mSystemSettings.EnableLocalMapping)
        mLocalMapper->setOutputWrapper(outputWrappers);
    if (mSystemSettings.EnableLoopClosure)
        mLoopCloser->setOutputWrapper(outputWrappers);
}

void 
Mapper::stopMapper()
{
    if (mSystemSettings.EnableLocalMapping) mLocalMapper->stopLocalMapping();
    if (mSystemSettings.EnableLoopClosure) mLoopCloser->stopLoopCloser();
}

}
