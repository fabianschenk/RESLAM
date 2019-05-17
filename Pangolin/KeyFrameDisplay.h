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

#undef Success
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <sstream>
#include <fstream>

#include "../../IOWrapper/DataStructures.h"

namespace RESLAM
{
class CameraMatrix;
class FrameData;
class FrameHeader;

namespace IOWrap
{

template<int ppp>
struct InputPointSparse
{
	float u;
	float v;
    float idepth;
	float idepth_hessian;
	float relObsBaseline;
	int numGoodRes;
    unsigned char color[ppp];
	unsigned char status;
};

struct MyVertex
{
	float point[3];
	unsigned char color[4];
};

// stores a pointcloud associated to a Keyframe.
class KeyFrameDisplay
{

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	KeyFrameDisplay();
	~KeyFrameDisplay();

	// copies points from KF over to internal buffer,
	// keeping some additional information so we can render it differently.
	void setFromKF(const FrameData * const fh, const CameraMatrix& HCalib);
	// copies points from KF over to internal buffer,
	// keeping some additional information so we can render it differently.
    void setFromF(const FrameHeader& frame, const CameraMatrix& HCalib);

	// copies & filters internal data to GL buffer for rendering. if nothing to do: does nothing.
	bool refreshPC(bool canRefresh, float scaledTH, float absTH, int mode, float minBS, int sparsity);

	// renders cam & pointcloud.
	void drawCam(float lineWidth = 1, float* color = 0, float sizeFactor=1, const size_t idx = 0);
	void drawCamKF(float lineWidth = 1, float* color = 0, float sizeFactor=1, const size_t idx = 0);
	void drawPC(float pointSize);

	size_t frameId; ///< frame Id
	size_t kfId; ///< keyframe Id
	bool active;
    SE3Pose camToWorld;

	bool isInFernDB = false; ///< true if the keyframe is added to the database!
	bool isInLoop = false;

    inline bool operator < (const KeyFrameDisplay& other) const
    {
        return (frameId < other.frameId);
    }

private:
	float fx,fy,cx,cy;
	float fxi,fyi,cxi,cyi;
	int width, height;

	float my_scaledTH, my_absTH, my_scale;
	int my_sparsifyFactor;
	int my_displayMode;
	float my_minRelBS;
	bool needRefresh;


    size_t numSparsePoints;
    size_t numSparseBufferSize;
    InputPointSparse<NUMBER_OF_VIEWER_COLORS>* mOriginalInputSparse;

	bool bufferValid;
	int numGLBufferPoints;
	int numGLBufferGoodPoints;
	pangolin::GlBuffer vertexBuffer;
	pangolin::GlBuffer colorBuffer;
};

}
}

