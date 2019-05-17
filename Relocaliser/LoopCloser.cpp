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

#include "LoopCloser.h"
#include "../Utils/Logging.h"
#include "../System/Tracker.h"
#include "../System/Mapper.h"
#include "types.h"
#include "../Utils/timer.h"
#include "../Utils/utility_functions.h"
#include "../System/SystemSettings.h"
#include "FernRelocLib/fernrelocaliser.h"
namespace RESLAM
{

size_t
LoopCloser::getNumberOfFramesInFernDb() const 
{ 
    return mFernRelocaliser->getNumberOfFramesInFernDb(); 
}

LoopCloser::LoopCloser(const SystemSettings& settings, FernRelocLib::FernRelocaliser* fernRelocaliser, Mapper* globalMapperPtr, Tracker* trackerPtr):
            mCeresLoopCloser(settings), mSystemSettings(settings), mFernRelocaliser(fernRelocaliser), mGlobalMapperPtr(globalMapperPtr), mTrackerPtr(trackerPtr)
{
    mFlagLCIsRunning = !mSystemSettings.LoopClosureLinearProcessing;
    
    if (!mSystemSettings.LoopClosureLinearProcessing)
    {
        lcThread = std::thread(&LoopCloser::loopCloserThread, this);
    }
}

LoopCloser::~LoopCloser()
{
    I3D_LOG(i3d::info) << "Closing the loop: ~LoopCloser";
    if (!mSystemSettings.LoopClosureLinearProcessing) lcThread.join();
    constexpr bool StopLoopCloserThread{false};
    setFlagLoopCloserRunning(StopLoopCloserThread);
    I3D_LOG(i3d::info) << "Closed the loop thread: ~LoopCloser";
}

void
LoopCloser::stopLoopCloser()
{
    if (!mSystemSettings.LoopClosureLinearProcessing) setFlagLoopCloserRunning(false);
}

void
LoopCloser::queueKeyFrame(FrameData* frameData)
{
    I3D_LOG(i3d::info) << "queueKeyFrame Adding frame: " << frameData->mKeyFrameId;
    {
        std::lock_guard<std::mutex> l(mLCQueueMutex);
        mLCQueue.emplace(frameData);
    }
    mLCQueueCV.notify_all(); //notify all that a new frame is available
}

void
LoopCloser::loopCloserLinear(FrameData* newFrame)
{
    auto distance = FLOAT_INF;
    constexpr bool HarvestKeyframes{true};
    constexpr int InvalidNearestNeighbor{-1};
    int nnNeighbor = InvalidNearestNeighbor;

    I3D_LOG(i3d::info) << "nnNeighbor: " << nnNeighbor << " distance: " << distance << " newFrame: " << newFrame->mKeyFrameId;
    const auto startTime = Timer::getTime();
    const bool ret = mFernRelocaliser->ProcessFrame(newFrame,nOfKeyFramesInDB,nnNeighbor,distance, HarvestKeyframes,mSystemSettings.FernDatabaseAlwaysAddKf);
    const auto endTime = Timer::getTime();
    I3D_LOG(i3d::info) << "Time for frame processing: " << Timer::getTimeDiffMiS(endTime,startTime);
    if (ret && nnNeighbor == InvalidNearestNeighbor)
    {
        I3D_LOG(i3d::info) << "Inserted keyframe into DB " << distance << " " << nnNeighbor << " "
                            << nOfKeyFramesInDB << " insertedKfId: " << newFrame->mKeyFrameId << " " << ret;
        nOfKeyFramesInDB++;
    }
    else
    {
        if (nnNeighbor == InvalidNearestNeighbor)
        {
            I3D_LOG(i3d::info) << "Too similar to be added but not a valid LC candidate insertedKfId: " << newFrame->mKeyFrameId << " dist: " << distance;
        }
        else
        {
            auto* candidateFrame = mFernRelocaliser->mFernKeyframeDb[nnNeighbor].frameDataPtr;
            //Don't close loops between very close keyframes
            if (candidateFrame->mKeyFrameId+mSystemSettings.LoopClosureMinDistBetweenFrames < newFrame->mKeyFrameId)
            {
                if (mSystemSettings.LoopClosureDoNotCloseLoopsIfClosedLastN > 0 && !mEdgeConnections.empty())
                {
                    I3D_LOG(i3d::info) << "Checking connections!";
                    const auto& lastEdge = mEdgeConnections.back();
                    if (absDiffSizeT(lastEdge.i,newFrame->mKeyFrameId) < mSystemSettings.LoopClosureDoNotCloseLoopsIfClosedLastN ||
                        absDiffSizeT(lastEdge.j,newFrame->mKeyFrameId) < mSystemSettings.LoopClosureDoNotCloseLoopsIfClosedLastN)
                    // if (((lastEdge.i > newFrame->mKeyFrameId ? lastEdge.i - newFrame->mKeyFrameId : newFrame->mKeyFrameId - lastEdge.i) < mSystemSettings.LoopClosureDoNotCloseLoopsIfClosedLastN) ||
                        // ((lastEdge.j > newFrame->mKeyFrameId ? lastEdge.j - newFrame->mKeyFrameId : newFrame->mKeyFrameId - lastEdge.j) < mSystemSettings.LoopClosureDoNotCloseLoopsIfClosedLastN))
                    {
                        I3D_LOG(i3d::info) << "Not closing loop because closed one in the last " << mSystemSettings.LoopClosureDoNotCloseLoopsIfClosedLastN 
                                           << " keyframes with : " << lastEdge.i << "/" <<lastEdge.j << " kfId: " << newFrame->mKeyFrameId;
                        return;
                    }
                }
                I3D_LOG(i3d::info) << "Not inserted keyframe " << newFrame->mKeyFrameId << " into DB " <<  distance << " " << nnNeighbor;
                FrameData* candidateFrame = mFernRelocaliser->convertDbId2KeyFrameId(nnNeighbor);
                I3D_LOG(i3d::info) << "Loop Closure: Neighbor id db: " << nnNeighbor<< " and real id("
                                   << candidateFrame->mKeyFrameId <<") "<< "current kf: " << newFrame->mKeyFrameId << "("<< newFrame->mFrameHeader->mFrameId<<")";
                if (checkPotentialLoop(*candidateFrame,*newFrame))
                {
                    newFrame->setFlagIsInLoop(true);
                }
                else
                {
                    I3D_LOG(i3d::info) << "Not a valid loop, adding keyframe " << newFrame->mKeyFrameId << " to DB";
                    mFernRelocaliser->insertFrame(newFrame);
                }
            }
            // AVOID CLOSE KEYFRAMES IN DISTANCE COMPUTATION!!!
            else
            {
                I3D_LOG(i3d::info) << "No loop because too close!" << newFrame->mKeyFrameId << " with: " << distance;
            }
        }
    }
}

void
LoopCloser::loopCloserThread()
{

    I3D_LOG(i3d::info) << "Started LoopCloser thread!";
    while (isLoopCloserRunning())
    {
        std::unique_lock<std::mutex> l(mLCQueueMutex);
        mLCQueueCV.wait(l,[this]{ return !mLCQueue.empty(); });

        FrameData* newFrame = std::move(mLCQueue.front());
        mLCQueue.pop();
        loopCloserLinear(newFrame);
    }
}

/**
 * 
 * Computes the relative pose between a candidate and the current KF
 * 
 */ 

bool 
LoopCloser::optimizeLoopClosure(SE3Pose& T_l_c, const FrameData& loopCandidate, const FrameData& currFrame)
{
    I3D_LOG(i3d::info) << "Optimizing refFrame: " << loopCandidate.mKeyFrameId << " currFrame: " << currFrame.mKeyFrameId;
    //check how many edges are bad!
    ResidualInfo resInfo, resInfoEye;
    if (mSystemSettings.LoopClosureDebugShowTransformationImages)
        mTrackerPtr->reprojectToImage(loopCandidate,currFrame,T_l_c,"evaluateCostFunctionFastRT", false);
    const auto costRT = mTrackerPtr->evaluateCostFunctionFast(loopCandidate,currFrame,T_l_c,resInfo);
    if (mSystemSettings.LoopClosureDebugShowTransformationImages)
        mTrackerPtr->reprojectToImage(loopCandidate,currFrame,SE3Pose(),"evaluateCostFunctionFastEye",true);
    const auto costEye = mTrackerPtr->evaluateCostFunctionFast(loopCandidate,currFrame,SE3Pose(),resInfoEye);

    //Take the post with the lower cost
    SE3Pose T_l_c_init = (costRT < costEye ? T_l_c : SE3Pose());
    I3D_LOG(i3d::info) << "Choose R,T init! Init: " << T_l_c_init.matrix3x4() << " avg " << costRT/resInfo.nGoodEdges << " eye: " 
                       << costEye/resInfoEye.nGoodEdges << " edge: " << currFrame.mKeyFrameId << " to " << loopCandidate.mKeyFrameId;

    //TODO: find a better measure
    if ((resInfo.nGoodEdges > resInfo.nBadEdges*2 && costRT < costEye) || 
        (resInfoEye.nGoodEdges > resInfoEye.nBadEdges*2 && costRT >= costEye)) 
    {
        
        const auto trackerStatus = mTrackerPtr->trackFrames(loopCandidate,currFrame,T_l_c_init,resInfo);
        
        if (mSystemSettings.LoopClosureDebugShowTransformationImages)
            mTrackerPtr->reprojectToImage(loopCandidate,currFrame,T_l_c,"before rel opt",false);
        T_l_c = T_l_c_init;
        if (mSystemSettings.LoopClosureDebugShowTransformationImages)
            mTrackerPtr->reprojectToImage(loopCandidate,currFrame,T_l_c,"after rel opt",true);
        I3D_LOG(i3d::info) << "Tracked T_l_c: " << T_l_c.matrix3x4();
        
        return (trackerStatus == Tracker::TrackerStatus::Ok);
    }
    I3D_LOG(i3d::info) << "No loop found because " << resInfo.nGoodEdges << " < " << resInfo.nBadEdges << " * 2";
    return false;
}

/**
 * Assess the loop quality by histogram voting verification
 */
bool LoopCloser::assessLoopQuality(const FrameData& frameData, const KeyframePoseVector& keyFrames) const
{
    I3D_LOG(i3d::info) << " frameData.getEdgesGood().empty(): " << frameData.getEdgesGood().empty();
    assert(frameData.getEdgesGood().empty());
    return mTrackerPtr->histogramVotingVerification(keyFrames,frameData,mSystemSettings.LoopClosureHistVotingWeights);
}

/**
 * Visualizes the loop closure check
 */
void 
LoopCloser::visualizeAssessLoopQuality(const FrameData& frameData, const KeyframePoseVector& keyFrames, std::string title) const
{
    size_t nImages{0};
    for (const auto& frameToCheck : keyFrames)
    {
        //FIXME: Maybe also include the candidate frame
        mTrackerPtr->reprojectToImage((*frameToCheck.first),frameData,frameToCheck.second,title+std::to_string(nImages),false);
        nImages++;
    }
    cv::waitKey(0);
}

/**
 * Test if the candidate frame is even valid!
 */
bool LoopCloser::checkPotentialLoop(const FrameData& candidateFrame,const FrameData& currFrame)
{
    if (mSystemSettings.LoopClosureDebugShowTransformationImages)
    {
        cv::imshow("candidate"+std::to_string(candidateFrame.mFrameSet->mTimestamp),candidateFrame.returnRgbFullSize());
        cv::imshow("curr"+std::to_string(currFrame.mFrameSet->mTimestamp),currFrame.returnRgbFullSize());
    }
    SE3Pose T_cand_curr_init = mGlobalMapperPtr->computeRelativeTransformationTf1f2(candidateFrame,currFrame);//T_cand_curr;
    I3D_LOG(i3d::info) << "T_cand_curr_init: " << T_cand_curr_init.matrix3x4();
    bool loopClosureOk = optimizeLoopClosure(T_cand_curr_init,candidateFrame,currFrame);
    if (loopClosureOk)
    {
        auto assessFrames = mGlobalMapperPtr->getSurroundingFrames(candidateFrame, T_cand_curr_init, mSystemSettings.LoopClosureFramesToCheck);
        //Check if relative transformation is ok
        const bool loopAssessmentValid = assessLoopQuality(currFrame,assessFrames);
        if (mSystemSettings.LoopClosureDebugShowTransformationImages)
            visualizeAssessLoopQuality(currFrame,assessFrames,"loopAfterRelOpt");
        
        if (loopAssessmentValid)
        {
            I3D_LOG(i3d::info) << "Loop  assessment valid for " << candidateFrame.mKeyFrameId << " and " << currFrame.mKeyFrameId
                               << " frameIds: " << candidateFrame.mFrameHeader->mFrameId << " and " << currFrame.mFrameHeader->mFrameId;
            CeresConstraint newestEdgeConnection(T_cand_curr_init, currFrame.mKeyFrameId, candidateFrame.mKeyFrameId);
            CeresPoseVector ceresVectorPoses;
            RelPoseConstraints ceresConstraints;
            //Now create a new pose graph
            //Add the loop constraints that have already been established
            for (const auto& edgeConnection : mEdgeConnections)
            {    
                //Add the same connection "N" times 
                //This is similar to giving it a weight N times higher than relative constraints
                for (size_t idx = 0; idx < mSystemSettings.LoopClosureEdgeWeight;++idx)
                    ceresConstraints.push_back(edgeConnection);
            }
            //Add the newest constraint
            for (size_t idx = 0; idx < mSystemSettings.LoopClosureEdgeWeight;++idx)
                ceresConstraints.push_back(newestEdgeConnection);

            //Here, we should wait till map optimization is done!
            //Copy the poses and the relative constraints
            mGlobalMapperPtr->copyKeyframePosesAndConstraints(ceresVectorPoses,ceresConstraints);

            for (const auto& c :ceresConstraints)
            {
                I3D_LOG(i3d::info) << "Constraint: " << c.i << "/" << c.j;
            }

            //Count the numbers of constraints for each frame
            if (mSystemSettings.LoopClosureDebug)
            {
                std::vector<size_t> hist(ceresVectorPoses.size());
                std::fill(hist.begin(),hist.end(),0);
                for (const auto& c : ceresConstraints)
                {
                    ++hist[c.i];
                    ++hist[c.j];
                }

                for (size_t idx = 0; idx < hist.size(); ++idx)
                {
                    I3D_LOG(i3d::info) << "idx: " << idx << ": " << hist[idx];
                }
            }

            //if loop got closed successfully, we add the connection to the edges!
            if (optimizeLoopClosureCeres(ceresVectorPoses,ceresConstraints))
            {
                //We need T_cand_curr
                const SE3Pose T_W_curr = ceresVectorPoses[currFrame.mKeyFrameId].returnPose();
                for (auto& kfAndPose : assessFrames)
                {
                    //Note that we have to go through the ref. frame since it is the only updated one
                    const SE3Pose T_ref_W = ceresVectorPoses[kfAndPose.first->mFrameHeader->getRefFrame()->mKeyFrameId].returnPose().inverse();
                    const SE3Pose T_cand_ref = kfAndPose.first->mFrameHeader->getRefToCam();
                    // const SE3Pose T_ref_cand = T_cam_ref * T_ref_W;
                    const SE3Pose T_cand_W = T_cand_ref * T_ref_W; //ceresVectorPoses[kfAndPose.first->mKeyFrameId].returnPose().inverse();
                    I3D_LOG(i3d::info) << "kfAndPose.second before: " << kfAndPose.second.matrix3x4() << " cand: " << kfAndPose.first->mKeyFrameId << " curr: " << currFrame.mKeyFrameId;
                    kfAndPose.second = T_cand_W * T_W_curr;
                    I3D_LOG(i3d::info) << "kfAndPose.second after: " << kfAndPose.second.matrix3x4();
                }
                //now, we have to do a final loop closure check to be sure!
                const bool isOkLoop = assessLoopQuality(currFrame,assessFrames);
                if (mSystemSettings.LoopClosureDebugShowTransformationImages)
                    visualizeAssessLoopQuality(currFrame,assessFrames,"loopAfterCeres");
                I3D_LOG(i3d::info) << (isOkLoop ? "LOOP IS OK" : "LOOP IS NOT OK");
                loopClosureOk = isOkLoop;
                if (isOkLoop)
                {
                    mEdgeConnections.push_back(newestEdgeConnection);
                    mGlobalMapperPtr->updatePosesAfterLoopClosure(ceresVectorPoses);
                    addFrameConstraintToGraph(newestEdgeConnection);
                    for(auto* ow : mOutputWrapper)
                        ow->publishEdgeGraph(mEdgeConnectivityGraph);
                    cv::destroyAllWindows();
                    return true;
                }
            }
        }
        else
        {
            I3D_LOG(i3d::info) << "Loop  assessment not valid for " << candidateFrame.mKeyFrameId << " and " << currFrame.mKeyFrameId;// << " with distance: " << distances <<"!";
        }
    }
    else
    {
        I3D_LOG(i3d::error) << "Bad loop closure " << currFrame.mKeyFrameId << " and " << candidateFrame.mKeyFrameId;// << mKeyframes_lc.size()-1 << "( " <<kfNewest->frameId <<" ) at " << nnNeighbors[0] << " with distance: " << distances[0] <<"!";
    }

    //If we get here, loop is not valid
    //Thus, we try to add it to the db
    cv::destroyAllWindows();
    return false;
}

void
LoopCloser::addFrameConstraintToGraph(const CeresConstraint& c) //const size_t frameId1, const size_t frameId2)
{
    addFrameConstraintToGraph(c.i,c.j);
}

void
LoopCloser::publishGraph()
{
    for(auto* ow : mOutputWrapper)
        ow->publishEdgeGraph(mEdgeConnectivityGraph);
}

void
LoopCloser::addFrameConstraintToGraph(const size_t frameId1, const size_t frameId2)
{
    mEdgeConnectivityGraph[(((uint64_t)frameId1) << 32) + ((uint64_t)frameId2)] = Vec2i(1,1);
    mEdgeConnectivityGraph[(((uint64_t)frameId2) << 32) + ((uint64_t)frameId1)] = Vec2i(1,1);
}

bool 
LoopCloser::optimizeLoopClosureCeres(CeresPoseVector& ceresVectorPoses, const RelPoseConstraints& ceresConstraints)//, const CeresConstraint& newestEdgeConnection)
{
    ceres::Problem problem;
    mCeresLoopCloser.BuildOptimizationProblem(ceresVectorPoses,ceresConstraints,&problem);
    const bool isUsable = mCeresLoopCloser.SolveOptimizationProblem(&problem);
    I3D_LOG(i3d::info) << "Solution is " << (isUsable? " is usable! " : " is not usable");
    return true;
}

void
LoopCloser::resetLoopCloser(const size_t deleteTillKfId)
{
    for (const auto& c : mEdgeConnections)
    {
        I3D_LOG(i3d::info) << "c bef: " << c.i << "/" << c.j;
    }
    //delete all the constraints that will be removed
    mEdgeConnections.erase( std::remove_if(mEdgeConnections.begin(), mEdgeConnections.end(),
                            [id=deleteTillKfId](const CeresConstraint& c){ return (c.i > id || c.j > id); }),
                            mEdgeConnections.end());
    for (const auto& c : mEdgeConnections)
    {
        I3D_LOG(i3d::info) << "c away: " << c.i << "/" << c.j;
    }
    //readd them
    for (const auto& c : mEdgeConnections)
        addFrameConstraintToGraph(c);
}
}