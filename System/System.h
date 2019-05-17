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
#include <vector>
#include <thread>

#include "SystemSettings.h"
#include "../IOWrapper/Input.h"
#include "Mapper.h"

namespace RESLAM
{
class CameraMatrix;
class Tracker;
class System
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    System(const std::string& settingsFileName, const std::string& datasetFileName);
    ~System();
    //starts RESLAM
    bool startSystem();
    //Stops the system and sends stop signals to all the depending threads,...
    void stopSystem();
    void addOutputWrapper(const std::vector<IOWrap::Output3DWrapper*> outputWrappers)
    {
        std::copy(outputWrappers.begin(),outputWrappers.end(),std::back_inserter(mOutputWrapper));
        //Forward also to wrapper
        mMapper.addOutputWrapper(outputWrappers);
    }

    auto& getOutputWrapper() { return mOutputWrapper;}
    const auto& getOutputWrapper() const { return mOutputWrapper;}
    const SystemSettings& returnSystemSettings() const { return mSystemSettings; }
    
private:
    void processFrame(std::unique_ptr<FrameData> newFrameData, std::unique_ptr<FrameHeader> newFrameHeader);
    void initSystem(FrameData* newFrameData);
    ///typically called after the second tracking lost!
    void setToRelocalisationMode();
    void relocaliseFrame(FrameData* newFrameData);
    void trackNewestFrame(std::unique_ptr<FrameData> newFrameData, std::unique_ptr<FrameHeader> fh);
    void relocalisation(std::unique_ptr<FrameData> newFrameData, std::unique_ptr<FrameHeader> fh);
    void printSLAMReport();
    enum class SystemStatus
    {
        Tracking,       //Everyhting normal
        Relocalisation, //relocalisation
        View,           //IO finished and view only
        Init            //Start up
    };
    System::SystemStatus mSystemStatus;
    SystemSettings mSystemSettings;
    Input mInputWrapper;
    Mapper mMapper; //The global mapper
    PoseVector mGtPoses; //The gt poses to compare the trajectories
    std::unique_ptr<CameraMatrix> mCameraMatrix; // The camera matrix
    std::unique_ptr<Tracker> mTracker;
    std::vector<IOWrap::Output3DWrapper*> mOutputWrapper;
    bool mLostInLastFrame; ///< true if tracking was lost. We only are lost if we have two tracking losses in a row!
    size_t mNTrackingLoss{0};
};
}
