/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once
#include "vector"
#include <math.h>
#include "../config/Defines.h"

namespace RESLAM
{
class CameraMatrix;
class EdgeFrameResidual;
class FrameData;
class CoarseDistanceMap {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	CoarseDistanceMap(int w, int h);
	~CoarseDistanceMap();

	void makeDistanceMap(const std::vector<FrameData*>& frameHessians, FrameData* frame);
    void makeK(const CameraMatrix& HCalib);

	float* fwdWarpedIDDistFinal;
	void addIntoDistFinal(const int u, const int v);
private:

    EdgeFrameResidual** coarseProjectionGrid;
	int* coarseProjectionGridNum;
	Vec2i* bfsList1;
	Vec2i* bfsList2;

    int pyrLevelsUsed;
	void growDistBFS(int bfsNum);

	Mat33f K[PyramidLevels];
	Mat33f Ki[PyramidLevels];

	//focal lengths the inverse
	std::array<float,PyramidLevels> fx,fy,fxi,fyi;//[PyramidLevels];
	//focal lengths the inverse
	std::array<float,PyramidLevels> cx,cy,cxi,cyi;//[PyramidLevels];
	//width and height
	std::array<int,PyramidLevels> w,h;
};
}