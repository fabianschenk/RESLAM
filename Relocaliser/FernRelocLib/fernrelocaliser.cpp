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
#include "fernrelocaliser.h"
#include "FernConservatory.h"
#include "RelocDatabase.h"

#include "../../IOWrapper/DataStructures.h"
#include "../../Utils/utility_functions.h"
#include "../../Utils/timer.h"
#include "../../System/SystemSettings.h"
namespace RESLAM
{
namespace FernRelocLib
{
inline void 
filterSubsampleWithHoles(float *imageData_out, const int x, const int y, const int newWidth, const float *imageData_in, const int origWidth)
{
    const int src_pos_x = x * 2, src_pos_y = y * 2;
    float pixel_out = 0.0f, pixel_in, no_good_pixels = 0.0f;

    pixel_in = imageData_in[(src_pos_x + 0) + (src_pos_y + 0) * origWidth];
    if (pixel_in > 0.0f) { pixel_out += pixel_in; no_good_pixels++; }

    pixel_in = imageData_in[(src_pos_x + 1) + (src_pos_y + 0) * origWidth];
    if (pixel_in > 0.0f) { pixel_out += pixel_in; no_good_pixels++; }

    pixel_in = imageData_in[(src_pos_x + 0) + (src_pos_y + 1) * origWidth];
    if (pixel_in > 0.0f) { pixel_out += pixel_in; no_good_pixels++; }

    pixel_in = imageData_in[(src_pos_x + 1) + (src_pos_y + 1) * origWidth];
    if (pixel_in > 0.0f) { pixel_out += pixel_in; no_good_pixels++; }

    if (no_good_pixels > 0) pixel_out /= no_good_pixels;

    imageData_out[x + y * newWidth] = pixel_out;
}

//For depth, we need to deal with the special case, when depth is invalid (is equal to 0).
//this was partly taken from InfiniTAM (https://github.com/victorprad/InfiniTAM)
void 
FilterSubsampleWithHoles(cv::Mat& out, const cv::Mat& in)
{
    if (!(out.rows == int(in.rows/2) && out.cols == int(in.cols/2))) return;

    const float *imageData_in = (float*)in.data;
    float *imageData_out = (float*) out.data;

    for (int y = 0; y < out.rows; y++) for (int x = 0; x < out.cols; x++)
        filterSubsampleWithHoles(imageData_out, x, y, out.cols, imageData_in, in.cols);
}

FernRelocaliser::~FernRelocaliser() = default;

inline int 
FernRelocaliser::returnCodeLength() const
{
	return mEncoding->getNumFerns();
} 

FrameData* 
FernRelocaliser::convertDbId2KeyFrameId(const size_t idInDb) const
{
	return (idInDb < mFernKeyframeDb.size() ? mFernKeyframeDb.at(idInDb).frameDataPtr : nullptr);
}

int
FernRelocaliser::reAddFrames(const std::vector<std::unique_ptr<FrameData>>& keyFrames)
{
	size_t nAddedFrames{0};
	for (const auto& frameData : keyFrames)
	{
		if (frameData->isInFernDb())
		{
			const auto ret = mRelocDatabase->addEntry(frameData->getFernCode());
			assert(ret == mFernKeyframeDb.size());
			mFernKeyframeDb.emplace_back(frameData.get(),ret);
			nAddedFrames++;
		}
	}
	return nAddedFrames;
}

int 
FernRelocaliser::insertFrame(FrameData* frameData)//const char* code, FrameData* frameData)
{
	I3D_LOG(i3d::info) << "Inserting: " << frameData->mKeyFrameId;// keyFrameId;
	frameData->setFlagIsInKfDb(true);
	const char* code = frameData->getFernCode();
	const auto ret = mRelocDatabase->addEntry(code);
	assert(ret == mFernKeyframeDb.size());
	mFernKeyframeDb.emplace_back(frameData,ret);
	return ret;
}

int
FernRelocaliser::similarFrameInDB(FrameData* inputFrame)
{
	computeCode(inputFrame);
	const char * code = inputFrame->getFernCode();
	const auto distToAllFrames = mRelocDatabase->computeDistancesToKeyframes(code);

	const auto matchIt = std::min_element(distToAllFrames.cbegin(),distToAllFrames.cend());
	if (matchIt != distToAllFrames.end())
		return std::distance(distToAllFrames.cbegin(),matchIt);
	constexpr int NoSimilarFrameFound{-1};
	return NoSimilarFrameFound;
}

bool 
FernRelocaliser::ProcessFrame(FrameData* inputFrame, int keyFrameId, int &nearestNeighbours, float &distances, bool harvestKeyframes, bool alwaysAddKeyframe)
{
	computeCode(inputFrame);
	const char * code = inputFrame->getFernCode();

	// prepare outputs
	int ret = -1;
	nearestNeighbours = -1;
	distances = -1;
	I3D_LOG(i3d::info) << "before findMostSimilar: " << nearestNeighbours << " " << distances;
	I3D_LOG(i3d::info) << "before code: " << code;// nearestNeighbours << " " << distances;

	bool addFrameToDatabase = true;
	// find similar frames
	if (!mFernKeyframeDb.empty())
	{
		const auto distToAllFrames = mRelocDatabase->computeDistancesToKeyframes(code);
		//Now, go through all the distances and look for two things
		//- The min dist to determine if we should insert a frame into the DB
		//- The min dist for potential candidates not including the last N keyframes (Note: that not all N are typically inserted into the DB)
		// float minDist = DOUBLE_INF_EVAL, minDistMatching = DOUBLE_INF_EVAL;

		float overallMin{FLOAT_INF};
		// I3D_LOG(i3d::info) << "inputFrame->mKeyFrameId: " << inputFrame->mKeyFrameId 
		// 				   << " mSystemSettings.LoopClosureMinDistBetweenFrames: " << mSystemSettings.LoopClosureMinDistBetweenFrames;
		
		//We only match if there are enough frames
		if (inputFrame->mKeyFrameId > mSystemSettings.LoopClosureMinDistBetweenFrames)
		{
			const auto startTime = Timer::getTime();
			const auto maxMatchingID = (inputFrame->mKeyFrameId > mSystemSettings.LoopClosureMinDistBetweenFrames ? 
										inputFrame->mKeyFrameId - mSystemSettings.LoopClosureMinDistBetweenFrames : 0);
			// I3D_LOG(i3d::info) << "mFernKeyframeDb: " << mFernKeyframeDb.size() << " maxMatchingId: " << maxMatchingID;
			const auto matchingEndIt = std::find_if(mFernKeyframeDb.crbegin(),mFernKeyframeDb.crend(),
										[maxMatchingID](const FernKeyframe& kf){
											return kf.frameDataPtr->mKeyFrameId <= maxMatchingID;} );
			
			// I3D_LOG(i3d::info) << "Found end at: " << std::distance(mFernKeyframeDb.cbegin(),matchingEndIt.base()) << " maxMatchingId: " << maxMatchingID;

			const auto newMatchEnd = distToAllFrames.cbegin() + std::distance(mFernKeyframeDb.cbegin(),matchingEndIt.base());

			// I3D_LOG(i3d::info) << "Searching std::min_element(distToAllFrames.cbegin(),newMatchEnd): " << std::distance(distToAllFrames.cbegin(),newMatchEnd);
			const auto matchIt = std::min_element(distToAllFrames.cbegin(),newMatchEnd);
			// I3D_LOG(i3d::info) << "Searching std::min_element(newMatchEnd,distToAllFrames.cend()): " << std::distance(newMatchEnd,distToAllFrames.cend());
			const auto restMinDistIt = std::min_element(newMatchEnd,distToAllFrames.cend());
			
			// const auto overallMin = (restMinDistIt != distToAllFrames.end() ? std::min(*matchIt,*restMinDistIt) : matchIt);
			overallMin = (restMinDistIt != distToAllFrames.end() ? std::min(*matchIt,*restMinDistIt) : * matchIt);
			// I3D_LOG(i3d::info) << "*matchIt: " << *matchIt << " mSystemSettings.LoopClosureKeyframeThreshold: " << mSystemSettings.LoopClosureKeyframeThreshold;
			if (*matchIt < mSystemSettings.LoopClosureKeyframeThreshold)
			{
				ret =  std::distance(distToAllFrames.cbegin(),matchIt);
				nearestNeighbours = ret;
				distances = *matchIt;
			}
			const auto endTime = Timer::getTime();
			I3D_LOG(i3d::info) << "Found matching at: " << std::distance(distToAllFrames.cbegin(),matchIt) << " with distance " << *matchIt
							   << "Found rest min at: " << std::distance(distToAllFrames.cbegin(),restMinDistIt) << " with distance " << *restMinDistIt
							   << " overall min: " << overallMin << " time: " << Timer::getTimeDiffMiS(startTime,endTime);

		}
		else //No matching, just check if we have to insert a frame
		{
			const auto restMinDistIt = std::min_element(distToAllFrames.cbegin(),distToAllFrames.cend());
			if (restMinDistIt != distToAllFrames.end())
				overallMin = *restMinDistIt;

		}		
		//Check if we already have enough similar frames
		if (overallMin < FernDBKfHarvestingThreshold) 
			addFrameToDatabase = false;
	}

	if (addFrameToDatabase)
	{
		ret = insertFrame(inputFrame);
	}

	I3D_LOG(i3d::info) << " ret: " << ret;
	return ret >= 0;
}

void
FernRelocaliser::computeCode(FrameData* inputFrame)
{
	cv::pyrDown(inputFrame->returnRgbFullSize(),processedColorImageHalf); //320x240
	cv::pyrDown(processedColorImageHalf,processedColorImageFourth); // 160x120
	cv::pyrDown(processedColorImageFourth,processedColorImageEigth);// 80x60
	cv::pyrDown(processedColorImageEigth,processedColorImageSixteenth); // 40x30
	char* code =  inputFrame->getFernCode();//mEncoding->getNumFerns());
    constexpr bool mUseDepthImage = false;
    //Maybe try to directly down-scale the image
	if (mUseDepthImage)
	{
		// downsample and preprocess image => processedImage1
		//if we process depth, then we have to be careful about the holes!
		processedDepthImageHalf.setTo(0);processedDepthImageFourth.setTo(0);
		processedDepthImageEigth.setTo(0);processedDepthImageSixteenth.setTo(0);
		cv::Mat depthImg = inputFrame->mFrameSet->depth;
		FilterSubsampleWithHoles(processedDepthImageHalf,depthImg);
		FilterSubsampleWithHoles(processedDepthImageFourth,processedDepthImageHalf);
		//depthImg -> is already at 160x120
		FilterSubsampleWithHoles(processedDepthImageEigth,depthImg) ;// 80x60
		FilterSubsampleWithHoles(processedDepthImageSixteenth,processedDepthImageEigth); // 40x30
		// compute code
		mEncoding->computeCodeRGBDepth(processedColorImageSixteenth, processedDepthImageSixteenth, code);
	}
	else
		mEncoding->computeCodeRGB(processedColorImageSixteenth,code);
}

void
FernRelocaliser::resetFernRelocaliser(const std::vector<std::unique_ptr<FrameData>>& kfData)//, const size_t deleteTillFrameId)
{
	I3D_LOG(i3d::info) << "Deleting all";// keyFrameId;

	//Reset complete database and readd the entries!
	mRelocDatabase->resetDatabase();
	mFernKeyframeDb.clear();
	//Now, readd everything
	reAddFrames(kfData);
}

void 
FernRelocaliser::SaveToDirectory(const std::string& outputDirectory)
{
	std::string configFilePath = outputDirectory + "config.txt";
	std::ofstream ofs(configFilePath.c_str());

	//TODO MAKE WORK WITH TEMPLATE - type should change?
	if (!ofs) throw std::runtime_error("Could not open " + configFilePath + " for reading");
	ofs << "type=rgb,levels=4,numFerns=" << mEncoding->getNumFerns() << ",numDecisionsPerFern=" 
		<< mEncoding->getNumDecisions() / 3 << ",harvestingThreshold=" << FernDBKfHarvestingThreshold;

	mEncoding->SaveToFile(outputDirectory + "ferns.txt");
	mRelocDatabase->SaveToFile(outputDirectory + "frames.txt");
}

void 
FernRelocaliser::LoadFromDirectory(const std::string& inputDirectory)
{
	std::string fernFilePath = inputDirectory + "ferns.txt";
	std::string frameCodeFilePath = inputDirectory + "frames.txt";
	std::string posesFilePath = inputDirectory + "poses.txt";

	if (!std::ifstream(fernFilePath.c_str())) throw std::runtime_error("unable to open " + fernFilePath);
	if (!std::ifstream(frameCodeFilePath.c_str())) throw std::runtime_error("unable to open " + frameCodeFilePath);
	if (!std::ifstream(posesFilePath.c_str())) throw std::runtime_error("unable to open " + posesFilePath);

	mEncoding->LoadFromFile(fernFilePath);
	mRelocDatabase->LoadFromFile(frameCodeFilePath);
}
FernRelocaliser::FernRelocaliser(const cv::Size2i& imgSize,const SystemSettings& settings) : mSystemSettings(settings)
    {
        mUseDepthImage = false;
        constexpr size_t Levels{4};
        //(1 << levels) = 32
        I3D_LOG(i3d::info) << "imgSize / (1 << levels): " << imgSize / (1 << Levels) << " height: " << imgSize.height << " width: " << imgSize.width;
        const cv::Vec2f range(0,255);
        mEncoding = std::make_unique<FernConservatory>(imgSize / (1 << Levels), range);
        I3D_LOG(i3d::info) << "Creating RelocDatabase!";
        mRelocDatabase = std::make_unique<RelocDatabase>();//numFerns, mEncoding->getNumCodes());
        processedColorImageHalf = cv::Mat(imgSize/2, CV_8UC3, cv::Scalar(0)); //320x240
        processedColorImageFourth = cv::Mat(imgSize/4, CV_8UC3, cv::Scalar(0)); //160x120
        processedColorImageEigth = cv::Mat(imgSize/8, CV_8UC3, cv::Scalar(0)); //80x60
        processedColorImageSixteenth = cv::Mat(imgSize/16, CV_8UC3, cv::Scalar(0)); //40x30

        processedDepthImageHalf = cv::Mat(imgSize/2, CV_32FC1, cv::Scalar(0)); //320x240
        processedDepthImageFourth = cv::Mat(imgSize/4, CV_32FC1, cv::Scalar(0)); //160x120
        processedDepthImageEigth = cv::Mat(imgSize/8, CV_32FC1, cv::Scalar(0)); //80x60
        processedDepthImageSixteenth = cv::Mat(imgSize/16, CV_32FC1, cv::Scalar(0)); //40x30
    }
}
}