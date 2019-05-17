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
#include <opencv2/core.hpp>
#include "System.h"
#include "../Utils/timer.h"
#include "../Utils/Logging.h"
#include "Tracker.h"

namespace RESLAM
{

SystemSettings::SystemSettings(const std::string& settingsFileName, const std::string& datasetFileName)
{
    //There are actually three cases:
    //dataset file points somewhere -> read from there
    //dataset file is empty -> try to find the config entry in the settings file
    //dataset file is empty -> try to read everything from settings file
    readSettings(settingsFileName);
    if (datasetFileName.empty())
    {
        //no config file found in settings
        if (DatasetConfigFileName.empty())
            DatasetConfigFileName = settingsFileName;
    }
    else
        DatasetConfigFileName = datasetFileName;

    if (!readSensorSettings(DatasetConfigFileName))
    {
        I3D_LOG(i3d::error) << "Couldn't read sensor settings!";
        exit(0);
    }
    readViewerSettings(settingsFileName);
}

size_t 
SystemSettings::readSizeT(const cv::FileStorage& fs, const std::string& str, const size_t defaultValue) const
{
    int tmpInt;
    cv::read(fs[str],tmpInt,defaultValue);
    if (tmpInt > 0) return static_cast<size_t>(tmpInt);
    return (tmpInt > 0 ? static_cast<size_t>(tmpInt) : 0);
}

bool 
SystemSettings::readSensorSettings(const std::string &sensorSettingsFileName)
{
    cv::FileStorage fs(sensorSettingsFileName,cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        I3D_LOG(i3d::error) << "Couldn't open settings file at location: " << sensorSettingsFileName;
        exit(0);
    }
    //image pyramid settings (camera matrix, resolutions,...)
    cv::read(fs["InputFormat"], InputFormat, 0);
    cv::read(fs["InputSensorType"], InputSensorType, 0);
    cv::read(fs["InputDepthScaleFactor"],InputDepthScaleFactor,1000.0f);
    cv::read(fs["IntrinsicsImgWidth"],IntrinsicsImgSize.width,640);
    cv::read(fs["IntrinsicsImgHeight"],IntrinsicsImgSize.height,480);
    cv::read(fs["IntrinsicsCamFx"],IntrinsicsCamFx,(IntrinsicsImgSize.width+IntrinsicsImgSize.height)/2);
    cv::read(fs["IntrinsicsCamFy"],IntrinsicsCamFy,(IntrinsicsImgSize.width+IntrinsicsImgSize.height)/2);
    cv::read(fs["IntrinsicsCamCx"],IntrinsicsCamCx,IntrinsicsImgSize.width/2);
    cv::read(fs["IntrinsicsCamCy"],IntrinsicsCamCy,IntrinsicsImgSize.height/2);
    cv::read(fs["InputStopAfterNFrames"],InputStopAfterNFrames,10000);
    
    InputDatasetFile = (std::string)fs["InputDatasetFile"];
    I3D_LOG(i3d::info) << "InputDatasetFile1: " << InputDatasetFile;
    if (InputDatasetFile.empty()) InputDatasetFile = "associate.txt";
    I3D_LOG(i3d::info) << "InputDatasetFile2: " << InputDatasetFile;
    InputDatasetFolder = (std::string)fs["InputDatasetFolder"];
    cv::read(fs["InputCannyEdgeTh1"],InputCannyEdgeTh1,80);
    cv::read(fs["InputCannyEdgeTh2"],InputCannyEdgeTh2,60);
    cv::read(fs["InputReadGT"],InputReadGT,false);
    cv::read(fs["InputDatasetConfigFileName"],DatasetConfigFileName,"");
    cv::read(fs["InputDepthMin"],InputDepthMin,0.1f);
    cv::read(fs["InputDepthMax"],InputDepthMax,5.2f);
    cv::read(fs["InputColorFormatRGB"],InputColorFormatRGB,true);
    cv::read(fs["InputSmoothEdgeImage"],InputSmoothEdgeImage,true);
    cv::read(fs["InputComputeGradientsForEdgeDetector"],InputComputeGradientsForEdgeDetector,false);

    //only needed to skip auto exposure artifacts
    InputSkipFirstNFrames = readSizeT(fs,"InputSkipFirstNFrames",0);
    I3D_LOG(i3d::info) << "InputSkipFirstNFrames: " << InputSkipFirstNFrames;
    InputReadNFrames = readSizeT(fs,"InputReadNFrames",10000);
    if (Input::intToSensorType(InputSensorType) == Input::SensorType::Dataset && OutputRecordImages)
    {
        I3D_LOG(i3d::info) << "Reading from dataset, deactivating image recording!";
    }
    I3D_LOG(i3d::info) << "InputDepthMin: " << InputDepthMin << " InputDepthMax: " << InputDepthMax;
    return true;
}

void 
SystemSettings::readSettings(const std::string &settingsFileName)
{
    cv::FileStorage fs(settingsFileName,cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        I3D_LOG(i3d::error) << "Couldn't open settings file at location: " << settingsFileName;
        exit(0);
    }

    cv::read(fs["OutputRecordImages"],OutputRecordImages,false);
    cv::read(fs["OutputPoses"],OutputPoses,true);
    OutputPoseFileFolder = (std::string)fs["OutputPoseFileFolder"];
    
    cv::read(fs["TrackerResidualHuberWeight"],TrackerResidualHuberWeight,0.3f);
    cv::read(fs["TrackerOpticalFlowTFactor"],TrackerOpticalFlowTFactor,0.1f);
    cv::read(fs["TrackerOpticalFlowRTFactor"],TrackerOpticalFlowRTFactor,0.15f);
    cv::read(fs["TrackerOpticalFlowThreshold"],TrackerOpticalFlowThreshold,1.0f);
    cv::read(fs["TrackerAvgResidualBeforeTrackingLoss"],TrackerAvgResidualBeforeTrackingLoss,2.5f);
    TrackerEvalLvlForInit = readSizeT(fs,"TrackerEvalLvlForInit",2);
    cv::read(fs["TrackerSkipResdiualsOnLowerLvls"],TrackerSkipResdiualsOnLowerLvls,false);
    cv::read(fs["TrackerUseEdgeWeight"],TrackerUseEdgeWeight,false);
    cv::read(fs["TrackerUseEdgeFilter"],TrackerUseEdgeFilter,true);
    cv::read(fs["TrackerHistDist"],TrackerHistDist,3);
    cv::read(fs["TrackerUseWeightsResidualsForErrorComputation"],TrackerUseWeightsResidualsForErrorComputation,true);
    cv::read(fs["TrackerTrackFromFrameToKf"],TrackerTrackFromFrameToKf,false);
    //Local Mapper
    cv::read(fs["EnableLocalMapping"],EnableLocalMapping,true);
    I3D_LOG(i3d::info) << "EnableLocalMapping: " << EnableLocalMapping;
    cv::read(fs["LocalMapperUseCoarseDistanceMap"],LocalMapperUseCoarseDistanceMap,true);
    cv::read(fs["LocalMapperOptimizeDepth"],LocalMapperOptimizeDepth,true); ///< Flag that triggers depth optimization
    cv::read(fs["LocalMapperOptimizeInitDepth"],LocalMapperOptimizeInitDepth,true); ///< Flag that triggers depth optimization
    cv::read(fs["LocalMapperConditionalDepthOptimization"],LocalMapperConditionalDepthOptimization,false); ///< Flag that triggers depth optimization
    cv::read(fs["LocalMapperAdaptMinActDist"],LocalMapperAdaptMinActDist,false);
    cv::read(fs["LocalMapperMaxDistForValidPixels"],LocalMapperMaxDistForValidPixels,5.0f); ///< Maximum distance in the distance transform
    cv::read(fs["LocalMapperMinDistInDistMapForValidPixels"],LocalMapperMinDistInDistMapForValidPixels,3.0f); ///< Maximum distance to other activated points
    cv::read(fs["LocalMapperShowResidualsInImage"],LocalMapperShowResidualsInImage,true); ///< Maximum distance to other activated points
    cv::read(fs["LocalMapperEdgeDistance"],LocalMapperEdgeDistance,5.0f);
    cv::read(fs["LocalMapperLinearProcessing"],LocalMapperLinearProcessing,0);
    cv::read(fs["LocalMapperDoMarginalize"],LocalMapperDoMarginalize,true);
    cv::read(fs["LocalMapperUseNewPoseUpdate"],LocalMapperUseNewPoseUpdate,true);
    cv::read(fs["SystemMultiThreading"],SystemMultiThreading,true);
    I3D_LOG(i3d::info) <<"SystemMultiThreading: "<<SystemMultiThreading;
    //Debug stuff
    cv::read(fs["TrackerShowIterationsDebug"],TrackerShowIterationsDebug,false);
    cv::read(fs["TrackerShowInitDebug"],TrackerShowInitDebug,false);
    cv::Vec4f trackerHistWeights;
    cv::read(fs["TrackerHistVotingWeights"],trackerHistWeights,cv::Vec4f(1,1,1.25,1.5));
    TrackerHistVotingWeights = Vec4f(trackerHistWeights[0],trackerHistWeights[1],trackerHistWeights[2],trackerHistWeights[3]);
    I3D_LOG(i3d::info) << "Read: " << trackerHistWeights << " into " << TrackerHistVotingWeights.transpose();
    //Loop Closure
    cv::read(fs["EnableLoopClosure"],EnableLoopClosure,true);
    LoopClosureFramesToCheck = readSizeT(fs,"LoopClosureFramesToCheck",3);
    LoopClosureMinDistBetweenFrames = readSizeT(fs,"LoopClosureMinDistBetweenFrames",7);
    LoopClosureEdgeWeight = readSizeT(fs,"LoopClosureEdgeWeight",3);
    LoopClosureDoNotCloseLoopsIfClosedLastN = readSizeT(fs,"LoopClosureDoNotCloseLoopsIfClosedLastN",10);
    cv::read(fs["LoopClosureUseBAConstraints"],LoopClosureUseBAConstraints,true);
    cv::read(fs["LoopClosureOnlyUseFernKF"],LoopClosureOnlyUseFernKF,true);
    cv::read(fs["LoopClosureFixWindowPoses"],LoopClosureFixWindowPoses,true);
    cv::read(fs["LoopClosureAddAdjacentConstraints"],LoopClosureAddAdjacentConstraints,false);
    cv::read(fs["LoopClosureDoBackCheck"],LoopClosureDoBackCheck,false);
    cv::read(fs["FernDatabaseAlwaysAddKf"],FernDatabaseAlwaysAddKf,false);
    cv::read(fs["LoopClosureLinearProcessing"],LoopClosureLinearProcessing,false);
    cv::read(fs["LoopClosureDebugShowTransformationImages"],LoopClosureDebugShowTransformationImages,false);
    cv::read(fs["LoopClosureDebug"],LoopClosureDebug,false);
    cv::read(fs["LoopClosureTrackerShowIterations"],LoopClosureTrackerShowIterations,false);
    cv::read(fs["LoopClosureKeyframeThreshold"],LoopClosureKeyframeThreshold,0.3);

    cv::Vec4f lcHistWeights;
    cv::read(fs["LoopClosureHistVotingWeights"],lcHistWeights,cv::Vec4f(1,1,1.25,1.5));
    LoopClosureHistVotingWeights = Vec4f(lcHistWeights[0],lcHistWeights[1],lcHistWeights[2],lcHistWeights[3]);

    LoopClosureNumberOfBAConstraints = readSizeT(fs,"LoopClosureNumberOfBAConstraints",3);
    //Relocaliser
    cv::read(fs["EnableRelocaliser"],EnableRelocaliser,false);
    RemoveNFramesAfterTrackLoss = readSizeT(fs, "RemoveNFramesAfterTrackLoss",20);
    RelocaliserMinNumberOfKeyframes = readSizeT(fs, "RelocaliserMinNumberOfKeyframes",20);
    fs.release();
}

void 
SystemSettings::readViewerSettings(const std::string &settingsFileName)
{
    cv::FileStorage fs(settingsFileName,cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        I3D_LOG(i3d::error) << "Couldn't open settings file at location: " << settingsFileName;
        exit(0);
    }
    ViewerPointSize = readSizeT(fs,"ViewerPointSize",1);
    fs.release();
}

System::System(const std::string &settingsFileName, const std::string& datasetFileName):
        mSystemStatus(SystemStatus::Init), mSystemSettings(settingsFileName,datasetFileName), mInputWrapper(mSystemSettings), mMapper(mSystemSettings),mLostInLastFrame(false)
{
    mTracker = std::make_unique<Tracker>(mSystemSettings);
    mCameraMatrix = std::make_unique<CameraMatrix>(mSystemSettings);
    mGtPoses = mInputWrapper.readGTTrajectory(); // readGTTrajectory
    LOG_THRESHOLD((mSystemSettings.SystemDebugTextOutput ? i3d::debug : i3d::info));
}

bool
System::startSystem()
{
    //We have to process a linear and a mult-thread pipeline
    //For now, we assume that at least the sensor and the ui work in a thread
    mInputWrapper.startInput();
    if (mSystemSettings.InputReadGT)
        mOutputWrapper[0]->addGtTrajectory(mGtPoses);
    //TODO: Replace by a condition
    while (mInputWrapper.inputIsActive()) 
    {
        I3D_LOG(i3d::info) << "input is still active!";
        auto newestFrameSet = mInputWrapper.getFirstFrame();
        I3D_LOG(i3d::info) << "Got frame!";
        if (newestFrameSet == nullptr)
        {
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
          continue;
        } 
        auto newFrameHeader = std::make_unique<FrameHeader>(newestFrameSet->mTimestamp,0);
        //Transfer ownership
        auto newFrameData = std::make_unique<FrameData>(newFrameHeader.get(),std::move(newestFrameSet),mSystemSettings);
        processFrame(std::move(newFrameData),std::move(newFrameHeader));
    }
    printSLAMReport();
    if (mSystemSettings.OutputPoses)
        mMapper.outputPoses();
    mInputWrapper.stopInput();
    I3D_LOG(i3d::info) << "Stopping SLAM system!";
    
    return true;
}

void
System::stopSystem()
{
    I3D_LOG(i3d::nothing) << "stopSystem!";
    mInputWrapper.stopInput();
    I3D_LOG(i3d::nothing) << "stopMapping!";
    mMapper.stopMapper();
}

void
System::printSLAMReport()
{
    I3D_LOG(i3d::info) << "--------------------------------- RESLAM Report ---------------------------------";
    I3D_LOG(i3d::info) << "Number of Frames: " << mMapper.getNbOfTotalFrames();
    I3D_LOG(i3d::info) << "Number of Keyframes: " << mMapper.getNbOfKeyFrames();
    if (mSystemSettings.EnableLoopClosure)
    {
        I3D_LOG(i3d::info) << "Number of Loops: " << mMapper.getNumberOfLoopClosures();
        I3D_LOG(i3d::info) << "Number of Frames in FernDB: " << mMapper.getNumberOfFramesInFernDb();
    }
    I3D_LOG(i3d::info) << "Number of Tracking Losses: " << mNTrackingLoss;
    I3D_LOG(i3d::info) << "-----------------------------------------------------------------------------";
}

void 
System::processFrame(std::unique_ptr<FrameData> newFrameData, std::unique_ptr<FrameHeader> newFrameHeader)
{
    //new frame
    switch (mSystemStatus)
    {
        case SystemStatus::Init: //Check if the init pose is really I in the frameHeader
                                    mMapper.addKeyFrameHeaderAndId(std::move(newFrameHeader));
                                    {
                                        const auto* ptrToLastKf = newFrameData.get();
                                        mMapper.addKeyFrameToAll(std::move(newFrameData));
                                        for (auto* ow : mOutputWrapper) ow->pushLiveFrame(ptrToLastKf);
                                    }
                                    mSystemStatus = SystemStatus::Tracking;
                                    break;
        case SystemStatus::Tracking:
                                    newFrameData->prepareForTracking(mSystemSettings,*mCameraMatrix);
                                    trackNewestFrame(std::move(newFrameData),std::move(newFrameHeader));
                                    break;
        case SystemStatus::Relocalisation: 
                                    relocalisation(std::move(newFrameData),std::move(newFrameHeader)); 
                                    break;
        //view and default are the same
        case SystemStatus::View:;
        default:break;
    }
}

void
System::trackNewestFrame(std::unique_ptr<FrameData> newFrameData, std::unique_ptr<FrameHeader> fh)
{
    static size_t nFramesTracker{0};
    //compute old poses
    SE3Pose T_ref_N;
    const FrameData* newestKeyframe = mMapper.getMostRecentKeyframe();
    const auto startT = Timer::getTime();
    PoseVector posesToTry = mMapper.computePosesToTry();
    const auto endT = Timer::getTime();
    I3D_LOG(i3d::info) << "Time for computing poses: " << Timer::getTimeDiffMiS(startT,endT);

    if (mSystemSettings.TrackerTrackFromFrameToKf)
    {
        for (auto& p : posesToTry)
        {
            p = p.inverse();
        }
    }
    const auto startTime = Timer::getTime();
    auto trackingStatus = (mSystemSettings.TrackerTrackFromFrameToKf ?
                mTracker->findInitAndTrackFrames(*newestKeyframe, *newFrameData, T_ref_N, posesToTry) :
                mTracker->findInitAndTrackFrames(*newFrameData, *newestKeyframe, T_ref_N, posesToTry));
    const auto endTime = Timer::getTime();
    I3D_LOG(i3d::info) << "Time for tracking: " << Timer::getTimeDiffMiS(startTime,endTime) << "/" << Timer::getTimeDiffMs(startTime,endTime);

    if (trackingStatus != Tracker::TrackerStatus::Lost)
    {
        //Note that the newestKeyframe is always the reference frame, no matter the tracking mode
        fh->setCamToRef(mSystemSettings.TrackerTrackFromFrameToKf ? T_ref_N : T_ref_N.inverse());
        fh->setRefFrame(newestKeyframe);
        mLostInLastFrame = false;
    }
    I3D_LOG(i3d::info) << "Tracking Status: " << Tracker::printTrackerStatus(trackingStatus) << " T_ref_N: " << T_ref_N.matrix3x4();
    nFramesTracker++;

    switch (trackingStatus) {
        case Tracker::TrackerStatus::NewKeyframe:
            mMapper.addKeyFrameHeaderAndId(std::move(fh));
            {
                const auto* ptrToLastKf = newFrameData.get();
                mMapper.addKeyFrameToAll(std::move(newFrameData));
                for (auto* ow : mOutputWrapper) ow->pushLiveFrame(ptrToLastKf);
            }
            break;
        case Tracker::TrackerStatus::Lost:
            I3D_LOG(i3d::error) << "Tracking LOST! nFramesTracker: " << nFramesTracker << " lost in last frame: " << mLostInLastFrame;
            if (mLostInLastFrame) setToRelocalisationMode();
            mLostInLastFrame = true;
            mNTrackingLoss++;
            break;
        case Tracker::TrackerStatus::Ok:
        default:
            mMapper.addNormalFrameHeaderAndId(std::move(fh));
            for (auto* ow : mOutputWrapper) ow->pushLiveFrame(newFrameData.get());
            break;
    }
}

void
System::setToRelocalisationMode()
{
    const bool resetSystem = mMapper.setToRelocalisationMode();
    mSystemStatus = (resetSystem ? SystemStatus::Init : SystemStatus::Relocalisation);
    if (resetSystem)
    {
        I3D_LOG(i3d::info) << "resetting system!";
    }
    else
    {
        I3D_LOG(i3d::info) << "relocalising!";
    }
}

void 
System::relocalisation(std::unique_ptr<FrameData> newFrameData, std::unique_ptr<FrameHeader> fh)//FrameData* newFrameData)
{
    //It's probably better to only compute the valid edges and align it with the DT of the candidate!
    newFrameData->computeValidEdgePixels();
    SE3Pose T_t_s;
    if (mMapper.relocaliseFrame(std::move(newFrameData),std::move(fh),T_t_s))
    {
        I3D_LOG(i3d::info) << "Relocalisation successful" << T_t_s.matrix3x4();
        mSystemStatus = SystemStatus::Tracking;
    }
}

System::~System() = default;
}
