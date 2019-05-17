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

/**
 * 
 * Some of the code is also part of InfiniTAM
 * Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM
 * 
 */
#pragma once
#include <memory>
#include <fstream>
#include <sophus/se3.hpp>
#include <opencv2/core/mat.hpp>
#include "../../Utils/Logging.h"

namespace RESLAM
{
class FrameData;
class SystemSettings;

namespace FernRelocLib
{
class FernConservatory;
class RelocDatabase;
class FernRelocaliser
{
private:

    std::unique_ptr<FernConservatory> mEncoding;
    std::unique_ptr<RelocDatabase> mRelocDatabase;

    //rgb images
    cv::Mat processedImage1, processedImage2;

    //depth images
    cv::Mat processedDepthImageHalf, processedDepthImageFourth,
            processedDepthImageEigth, processedDepthImageSixteenth;

    cv::Mat processedColorImageHalf, processedColorImageFourth, processedColorImageEigth, processedColorImageSixteenth;

    // I think a vector with id and pair<real_id,ptr> is enough!
    struct FernKeyframe
    {
        size_t dbId; //unique id in database
        FrameData* frameDataPtr; //ptr to frame data
        FernKeyframe(FrameData* frameData, size_t id):dbId(id), frameDataPtr(frameData){}
    };
    bool mUseDepthImage;
    const SystemSettings& mSystemSettings;
public:
    std::vector<FernKeyframe> mFernKeyframeDb; ///< Database that stores the id and ptr to the keyframe

    FernRelocaliser(const cv::Size2i& imgSize,const SystemSettings& settings);
    ~FernRelocaliser();
    
    int similarFrameInDB(FrameData* inputFrame);
    inline int returnCodeLength() const;    

    void computeCode(FrameData* inputFrame);

    bool ProcessFrame(FrameData* inputFrame, int keyFrameId, int &nearestNeighbours, float &distances, bool harvestKeyframes, bool alwaysAddKeyframe);
    int insertFrame(FrameData* frameData);//int keyFrameId)
    int reAddFrames(const std::vector<std::unique_ptr<FrameData>>& keyFrames);
    auto getNumberOfFramesInFernDb() const { return mFernKeyframeDb.size(); }
    FrameData* convertDbId2KeyFrameId(const size_t idInDb) const;
    void resetFernRelocaliser(const std::vector<std::unique_ptr<FrameData>>& kfData);
    void SaveToDirectory(const std::string& outputDirectory);
    void LoadFromDirectory(const std::string& inputDirectory);
};
}
}