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
#include <pangolin/pangolin.h>
#include <boost/thread.hpp>
#include <map>
#include <deque>
#include "../MinimalImage.h"
#include "../Output3DWrapper.h"



namespace RESLAM
{
class SystemSettings;
class FrameData;
class CameraMatrix;
class FrameHeader;


namespace IOWrap
{

class KeyFrameDisplay;

struct GraphConnection
{
	KeyFrameDisplay* from;
	KeyFrameDisplay* to;
	int fwdMarg, bwdMarg, fwdAct, bwdAct;
};

using GraphConnections = std::vector<GraphConnection,Eigen::aligned_allocator<GraphConnection>>;

class PangolinDSOViewer : public Output3DWrapper
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PangolinDSOViewer(int w, int h, const SystemSettings& config, bool startRunThread=true);
	virtual ~PangolinDSOViewer();
	void addGtTrajectory(const PoseVector& gtPoses);
	void run();
	void close();

	void addImageToDisplay(std::string name, MinimalImageB3* image);
	void clearAllImagesToDisplay();


	// ==================== Output3DWrapper Functionality ======================
    virtual void publishGraph(const ConnectivityMap &connectivity);
	virtual void publishEdgeGraph(const ConnectivityMap &edgeConnectivity);
	virtual void publishMargGraph(const ConnectivityMap &edgeConnectivity);
    virtual void publishKeyframes( std::vector<FrameData*> &frames, const CameraMatrix& HCalib);
    virtual void publishCamPose(const FrameHeader& frame,  const CameraMatrix& HCalib);
	virtual void publishNewKeyframe(FrameData* frameData, const CameraMatrix& camMat);
	virtual void updateKeyframePoses(std::vector<FrameHeader*> keyFrameHeaders);
	virtual void publishCompleteTrajectory(const std::vector<std::unique_ptr<FrameHeader>>& frameHeaders);


    virtual void pushLiveFrame(const FrameData * const image);
	virtual void pushDepthImage(MinimalImageB3* image);
    virtual bool needPushDepthImage();

	virtual void join();

	virtual void reset();

	virtual void deleteTillFrameId(size_t frameId);
private:

	bool needReset;
	void reset_internal();
	void drawConstraints();

	boost::thread runThread;
	bool running;
	int w,h;

	// images rendering
	boost::mutex openImagesMutex;
	MinimalImageB3* internalVideoImg;
	MinimalImageB3* internalKFImg;
	MinimalImageB3* internalResImg;
	bool videoImgChanged, kfImgChanged, resImgChanged;

	// 3D model rendering
	boost::mutex model3DMutex;
	std::unique_ptr<KeyFrameDisplay> currentCam;
	using KeyFrameDisplayVector = std::vector<std::unique_ptr<KeyFrameDisplay>, Eigen::aligned_allocator<KeyFrameDisplay>>;
	KeyFrameDisplayVector mKeyframes;
	std::map<int, KeyFrameDisplay*> keyframesByKFID;
	GraphConnections mConnections; ///< connections BA
	GraphConnections mEdgeConnections; ///< connections from LC
	GraphConnections mMargConnections; ///< connections from marg
	
	PoseVector mAllPoses;
	PoseVector mGtPoses;
	const SystemSettings& mSystemSettings;

	// render settings
	bool settings_showKFCameras;
	bool settings_showCurrentCamera;
	bool settings_showTrajectory;
	bool settings_showFullTrajectory;
	bool settings_showGtTrajectory;
	bool settings_showActiveConstraints;
	bool settings_showLoopConstraints;
	bool settings_showMargConstraints;
	bool settings_showAllConstraints;

	float settings_scaledVarTH;
	float settings_absVarTH;
	int settings_pointCloudMode;
	float settings_minRelBS;
	int settings_sparsity;
    float setting_minGradHistCut = 0.5;
    bool setting_fullResetRequested = false;
    bool setting_render_displayCoarseTrackingFull=false;
    bool setting_render_renderWindowFrames=true;
    bool setting_render_plotTrackingFull = false;
    bool setting_render_display3D = true;
    bool setting_render_displayResidual = true;
    bool setting_render_displayVideo = true;
    bool setting_render_displayDepth = true;
    bool disableAllDisplay = false;
    float setting_desiredImmatureDensity = 1500; // immature points per frame
    float setting_minPointsRemaining = 0.05;  // marg a frame if less than X% points remain.
    float setting_maxLogAffFacInWindow = 0.7; // marg a frame if factor between intensities to current frame is larger than 1/X or X.

	// timings
	struct timeval last_track;
	struct timeval last_map;

	std::deque<float> lastNTrackingMs;
	std::deque<float> lastNMappingMs;
};
}
}
