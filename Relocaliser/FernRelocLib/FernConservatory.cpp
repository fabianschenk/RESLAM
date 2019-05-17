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
#include <fstream>
#include "FernConservatory.h"
#include "../../Utils/Logging.h"

namespace RESLAM
{
namespace FernRelocLib
{
static float random_uniform01(void)
{
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}
FernConservatory::FernConservatory(const cv::Size2i &imgSize, const cv::Vec2f &bounds)
{
    I3D_LOG(i3d::info) << "Generating FernConservatory with: " << FernDBNumFerns << " " 
                      << FernDBNumDecisionsionsPerFern << " " << imgSize << " " << bounds 
                      << " " << FernDBNumDecisionsionsPerFern << " width:  " << imgSize.width 
                      << "x h:" << imgSize.height;
	constexpr size_t NumberOfFernTesters = FernDBNumFerns*FernDBNumDecisionsionsPerFern;
    mEncoders = new FernTester[NumberOfFernTesters];
	for (size_t f = 0; f < NumberOfFernTesters; ++f) {
        mEncoders[f].location.x = static_cast<int>(floorf(random_uniform01() * imgSize.width));
        mEncoders[f].location.y = static_cast<int>(floorf(random_uniform01() * imgSize.height));
        mEncoders[f].threshold = random_uniform01() * (bounds[1] - bounds[0]) + bounds[0];
	}
}

FernConservatory::~FernConservatory(void)
{
	delete[] mEncoders;
}

void FernConservatory::computeCodeDepth(const cv::Mat &img, char *codeFragments) const
{
	for (size_t f = 0; f < FernDBNumFerns; ++f)
	{
		codeFragments[f] = 0;
		for (size_t d = 0; d < FernDBNumDecisionsionsPerFern; ++d)
		{
			const FernTester *tester = &(mEncoders[f*FernDBNumDecisionsionsPerFern + d]);
            const float val = img.at<float>(tester->location.y,tester->location.x);
            codeFragments[f] |= ((val < tester->threshold) ? 0 : 1) << d;
		}
	}
}

void FernConservatory::computeCodeRGB(const cv::Mat &img, char *codeFragments) const
{
	for (size_t f = 0; f < FernDBNumFerns; ++f)
	{
		codeFragments[f] = 0;
		for (size_t d = 0; d < FernDBNumOfDecisions; ++d)
		{
			const FernTester *tester = &mEncoders[f * FernDBNumOfDecisions + d];
            const unsigned char tester_threshold = static_cast<unsigned char>(tester->threshold);
            const cv::Vec3b clrVec = img.at<cv::Vec3b>(tester->location.y,tester->location.x);
            for (int c = 0; c < 3; ++c)
                if (clrVec[c] > tester_threshold) codeFragments[f] |= 1 << ((FernDBNumOfChannels * d) + c);
		}
	}
}
void FernConservatory::computeCodeRGBDepth(const cv::Mat& imgRGB, const cv::Mat& imgDepth, char *codeFragments) const
{
    const int nChannels = 4;
    const int numDecisions = FernDBNumDecisionsionsPerFern / nChannels;
    for (size_t f = 0; f < FernDBNumFerns; ++f)
    {
        codeFragments[f] = 0;
        for (size_t d = 0; d < numDecisions; ++d)
        {
            const FernTester *tester = &(mEncoders[f*numDecisions + d]);
            const float val = imgDepth.at<float>(tester->location.y,tester->location.x);
            const cv::Vec3b clrVec = imgRGB.at<cv::Vec3b>(tester->location.y,tester->location.x);
            const unsigned char tester_threshold = static_cast<unsigned char>(tester->threshold);
            for (int c = 0; c < 3; ++c)
                if (clrVec[c] > tester_threshold) codeFragments[f] |= 1 << ((nChannels * d) + c);
            codeFragments[f] |= ((val < tester->threshold) ? 0 : 1) << ((nChannels * d) + 3);
        }
    }
}
void FernConservatory::SaveToFile(const std::string &fernsFileName)
{
	std::ofstream ofs(fernsFileName.c_str());

	if (!ofs) throw std::runtime_error("Could not open " + fernsFileName + " for reading");;

	for (size_t f = 0; f < FernDBNumFerns * FernDBNumDecisionsionsPerFern; ++f)
        ofs << mEncoders[f].location.x << ' ' << mEncoders[f].location.y << ' ' << mEncoders[f].threshold << '\n';
}

void FernConservatory::LoadFromFile(const std::string &fernsFileName)
{
	std::ifstream ifs(fernsFileName.c_str());
	if (!ifs) throw std::runtime_error("unable to load " + fernsFileName);

	for (size_t i = 0; i < FernDBNumFerns; i++)
	{
		for (size_t j = 0; j < FernDBNumDecisionsionsPerFern; j++)
		{
			FernTester &fernTester = mEncoders[i * FernDBNumDecisionsionsPerFern + j];
            ifs >> fernTester.location.x >> fernTester.location.y >> fernTester.threshold;
		}
	}
}
}
}