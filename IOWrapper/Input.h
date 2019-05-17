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
#include <memory>
#include <mutex>
#include <queue>
#include <fstream>

#include "DataStructures.h"

namespace RESLAM
{
class SystemSettings;
class Sensor
{
protected:
    size_t mNbReceivedImages{0};
    const SystemSettings& mSystemSettings;
    bool mFlagIsRunning;
public:
    explicit Sensor(const SystemSettings& settings):mSystemSettings(settings),mFlagIsRunning(true) {}
    bool isRunning() const { return mFlagIsRunning; }
    void stopSensor() { std::cout << "Stopping Sensor" << std::endl; mFlagIsRunning = false; }
    virtual bool getImages(cv::Mat& rgb,cv::Mat& depth, double& timestamp) = 0;
    virtual ~Sensor(){};
};

class InputDatasetReader : public Sensor
{
public:
    enum class InputFormat
    {
        TUM,
        BundleFusion,
        Files
    };
     explicit InputDatasetReader(const SystemSettings& settings);
    virtual ~InputDatasetReader();
    bool getImages(cv::Mat& rgb,cv::Mat& depth, double& timestamp);
private:
    const std::string mFileName;
    std::ifstream mFileList;
    InputFormat mInputFormat;
};


class Input
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        enum class SensorType
        {
            Dataset, // 0
            OrbbecAstra, // 1
            OrbbecAstraPro, // 2
            RealsenseZR300, // 3
            RealsenseD4XX // 4
        };

        static SensorType intToSensorType(int sensorType)
        {
            switch (sensorType) {
            case 0: return SensorType::Dataset;
            case 1: return SensorType::OrbbecAstra;
            case 2: return SensorType::OrbbecAstraPro;
            case 3: return SensorType::RealsenseZR300;
            case 4: return SensorType::RealsenseD4XX;
            default: return SensorType::Dataset;
            }
        }

         explicit Input(const SystemSettings& settings):mSystemSettings(settings) {}

        bool isFrameAvailable() const
        {
            std::lock_guard<std::mutex> lock(mInputFrameQueueMutex);
            return !mInputFrameQueue.empty();
        }
        
        std::unique_ptr<FrameSet> getFirstFrame()
        {
            std::lock_guard<std::mutex> lock(mInputFrameQueueMutex);
            if (mInputFrameQueue.empty()) return nullptr;
            auto nextFrame = std::move(mInputFrameQueue.front());
            mInputFrameQueue.pop();
            return std::move(nextFrame);
        }
        void startInput();
        void stopInput();
        bool inputIsActive() const 
        { 
            std::lock_guard<std::mutex> lock(mInputFrameQueueMutex);
            return mSensor->isRunning() || !mInputFrameQueue.empty(); 
        }
        PoseVector readGTTrajectory();
    private:
        void readImages();
        std::queue<std::unique_ptr<FrameSet>> mInputFrameQueue;
        mutable std::mutex mInputFrameQueueMutex;
        const SystemSettings& mSystemSettings;
        std::unique_ptr<Sensor> mSensor;
        size_t mNbOfImages = 0;
};
}
