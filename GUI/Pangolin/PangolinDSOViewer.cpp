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
#include "PangolinDSOViewer.h"
#include "KeyFrameDisplay.h"
#include "../../Utils/timer.h"
#include "../../Utils/Logging.h"
#include "../../System/SystemSettings.h"

namespace RESLAM
{
namespace IOWrap
{
PangolinDSOViewer::PangolinDSOViewer(int w, int h, const SystemSettings& config, bool startRunThread): mSystemSettings(config)
{
	this->w = w;
	this->h = h;
	running = true;
	{
		boost::unique_lock<boost::mutex> lk(openImagesMutex);
		internalVideoImg = new MinimalImageB3(w,h);
		internalKFImg = new MinimalImageB3(w,h);
		internalResImg = new MinimalImageB3(w,h);
		videoImgChanged=kfImgChanged=resImgChanged=true;

		internalVideoImg->setBlack();
		internalKFImg->setBlack();
		internalResImg->setBlack();
	}
	{
		currentCam = std::make_unique<KeyFrameDisplay>();
	}

	needReset = false;


    if(startRunThread)
        runThread = boost::thread(&PangolinDSOViewer::run, this);

}


PangolinDSOViewer::~PangolinDSOViewer()
{
	internalVideoImg->release();
	internalVideoImg = nullptr;
	internalKFImg->release();
	internalKFImg = nullptr;
	internalResImg->release();
	internalResImg = nullptr;

	I3D_LOG(i3d::info) << "~PangolinDSOViewer JOINED Pangolin thread!";
	close();
	I3D_LOG(i3d::info) << "close ~PangolinDSOViewer JOINED Pangolin thread!";
	runThread.join();
	I3D_LOG(i3d::info) << "runThread.join ~PangolinDSOViewer JOINED Pangolin thread!";
}

void
PangolinDSOViewer::addGtTrajectory(const PoseVector& gtPoses)
{
	std::copy(gtPoses.cbegin(),gtPoses.cend(),std::back_inserter(mGtPoses));
}


void PangolinDSOViewer::run()
{
	printf("START PANGOLIN!\n");
	constexpr int wUi{1500},hUi{1280};
	pangolin::CreateWindowAndBind("Main",2*wUi,2*hUi);
	const int UI_WIDTH = 180;

	glEnable(GL_DEPTH_TEST);

	// 3D visualization
	pangolin::OpenGlRenderState Visualization3D_camera(
		pangolin::ProjectionMatrix(w,h,400,400,w/2,h/2,0.1,1000),
		pangolin::ModelViewLookAt(-0,-5,-10, 0,0,0, pangolin::AxisNegY));

	pangolin::View& Visualization3D_display = pangolin::CreateDisplay()
		.SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -w/(float)h)
		.SetHandler(new pangolin::Handler3D(Visualization3D_camera));


	// 3 images
	pangolin::View& d_kfDepth = pangolin::Display("imgKFDepth").SetAspect(w/(float)h);

	pangolin::View& d_video = pangolin::Display("imgVideo").SetAspect(w/(float)h);

	pangolin::View& d_residual = pangolin::Display("imgResidual").SetAspect(w/(float)h);

	pangolin::GlTexture texKFDepth(w,h,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
	pangolin::GlTexture texVideo(w,h,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
	pangolin::GlTexture texResidual(w,h,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);


    pangolin::CreateDisplay()
		  .SetBounds(0.0, 0.3, pangolin::Attach::Pix(UI_WIDTH), 1.0)
		  .SetLayout(pangolin::LayoutEqual)
		  .AddDisplay(d_kfDepth)
		  .AddDisplay(d_video)
		  .AddDisplay(d_residual);

	// parameter reconfigure gui
	pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

	pangolin::Var<int> settings_pointCloudMode("ui.PC_mode",1,0,4,false);

	pangolin::Var<bool> settings_showKFCameras("ui.KFCam",true,true);
	pangolin::Var<bool> settings_showCurrentCamera("ui.CurrCam",true,true);
	pangolin::Var<bool> settings_showTrajectory("ui.Trajectory",true,true);
	pangolin::Var<bool> settings_showFullTrajectory("ui.FullTrajectory",false,true);
	pangolin::Var<bool> settings_showGtTrajectory("ui.ShowGtTrajectory",false,true);
	pangolin::Var<bool> settings_showActiveConstraints("ui.ActiveConst",true,true);
	pangolin::Var<bool> settings_showLoopConstraints("ui.LoopConst",true,true);
	pangolin::Var<bool> settings_showMargConstraints("ui.MargConst",false,true);
	pangolin::Var<bool> settings_showAllConstraints("ui.AllConst",false,true);


	pangolin::Var<bool> settings_show3D("ui.show3D",true,true);
	pangolin::Var<bool> settings_showLiveDepth("ui.showDepth",true,true);
	pangolin::Var<bool> settings_showLiveVideo("ui.showVideo",true,true);
    pangolin::Var<bool> settings_showLiveResidual("ui.showResidual",false,true);

	pangolin::Var<bool> settings_showFramesWindow("ui.showFramesWindow",false,true);
	pangolin::Var<bool> settings_showFullTracking("ui.showFullTracking",false,true);
	pangolin::Var<bool> settings_showCoarseTracking("ui.showCoarseTracking",false,true);


	pangolin::Var<int> settings_sparsity("ui.sparsity",1,1,20,false);
	pangolin::Var<double> settings_scaledVarTH("ui.relVarTH",0.001,1e-10,1e10, true);
	pangolin::Var<double> settings_absVarTH("ui.absVarTH",0.001,1e-10,1e10, true);
	pangolin::Var<double> settings_minRelBS("ui.minRelativeBS",0.1,0,1, false);


	pangolin::Var<bool> settings_resetButton("ui.Reset",false,false);


	// pangolin::Var<int> settings_nCandidates("ui.pointCandidates",setting_desiredImmatureDensity, 50,5000, false);
	// pangolin::Var<double> settings_kfFrequency("ui.kfFrequency",setting_kfGlobalWeight,0.1,3, false);
	// pangolin::Var<double> settings_gradHistAdd("ui.minGradAdd",setting_minGradHistAdd,0,15, false);

	// pangolin::Var<double> settings_trackFps("ui.Track fps",0,0,0,false);
	// pangolin::Var<double> settings_mapFps("ui.KF fps",0,0,0,false);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	// Default hooks for exiting (Esc) and fullscreen (tab).
	while( !pangolin::ShouldQuit() && running )
	{
		const auto startTime = Timer::getTime();
		// Clear entire screen
		//glClearColor(163.0f/255.0f, 163.0f/255.0f, 163.0f/255.0f,1.0f); //ligh grey
		//glClearColor(50.0f/255.0f, 50.0f/255.0f, 50.0f/255.0f,1.0f);
		// glClearColor(50.0f/255.0f, 50.0f/255.0f, 50.0f/255.0f,1.0f);

		glClearColor(1.0f,1.0f,1.0f,1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		if(setting_render_display3D)
		{
			// Activate efficiently by object
			Visualization3D_display.Activate(Visualization3D_camera);
			boost::unique_lock<boost::mutex> lk3d(model3DMutex);
			int refreshed=0;
			int idx = 0;
			for(auto& fh : mKeyframes)
			{
				float blue[3] = {0,0,1};
				
				if(this->settings_showKFCameras) fh->drawCamKF(1,blue,0.1,idx);
				idx++;
				refreshed =+ (int)(fh->refreshPC(refreshed < 10, this->settings_scaledVarTH, this->settings_absVarTH,
						this->settings_pointCloudMode, this->settings_minRelBS, this->settings_sparsity));
				fh->drawPC(mSystemSettings.ViewerPointSize);
			}
			if(this->settings_showCurrentCamera) currentCam->drawCam(2,0,0.2,idx);
			drawConstraints();
			lk3d.unlock();
		}



		openImagesMutex.lock();
		if(videoImgChanged) 	texVideo.Upload(internalVideoImg->data,GL_BGR,GL_UNSIGNED_BYTE);
		if(kfImgChanged) 		texKFDepth.Upload(internalKFImg->data,GL_BGR,GL_UNSIGNED_BYTE);
		if(resImgChanged) 		texResidual.Upload(internalResImg->data,GL_BGR,GL_UNSIGNED_BYTE);
		videoImgChanged=kfImgChanged=resImgChanged=false;
		openImagesMutex.unlock();


		// update fps counters
		// {
		// 	openImagesMutex.lock();
		// 	float sd=0;
		// 	for(float d : lastNMappingMs) sd+=d;
		// 	settings_mapFps=lastNMappingMs.size()*1000.0f / sd;
		// 	openImagesMutex.unlock();
		// }

		// {
		// 	model3DMutex.lock();
		// 	float sd=0;
		// 	for(float d : lastNTrackingMs) sd+=d;
		// 	settings_trackFps = lastNTrackingMs.size()*1000.0f / sd;
		// 	model3DMutex.unlock();
		// }

		if(setting_render_displayVideo)
		{
			d_video.Activate();
			glColor4f(1.0f,1.0f,1.0f,1.0f);
			texVideo.RenderToViewportFlipY();
		}

		if(setting_render_displayDepth)
		{
			d_kfDepth.Activate();
			glColor4f(1.0f,1.0f,1.0f,1.0f);
			texKFDepth.RenderToViewportFlipY();
		}

		if(setting_render_displayResidual)
		{
			d_residual.Activate();
			glColor4f(1.0f,1.0f,1.0f,1.0f);
			texResidual.RenderToViewportFlipY();
		}


	    // update parameters
	    this->settings_pointCloudMode = settings_pointCloudMode.Get();

	    this->settings_showActiveConstraints = settings_showActiveConstraints.Get();
		this->settings_showLoopConstraints = settings_showLoopConstraints.Get();
		this->settings_showMargConstraints = settings_showMargConstraints.Get();
	    this->settings_showAllConstraints = settings_showAllConstraints.Get();
	    this->settings_showCurrentCamera = settings_showCurrentCamera.Get();
	    this->settings_showKFCameras = settings_showKFCameras.Get();
	    this->settings_showTrajectory = settings_showTrajectory.Get();
	    this->settings_showFullTrajectory = settings_showFullTrajectory.Get();
		this->settings_showGtTrajectory = settings_showGtTrajectory.Get();
		setting_render_display3D = settings_show3D.Get();
		setting_render_displayDepth = settings_showLiveDepth.Get();
		setting_render_displayVideo =  settings_showLiveVideo.Get();
		setting_render_displayResidual = settings_showLiveResidual.Get();

		setting_render_renderWindowFrames = settings_showFramesWindow.Get();
		setting_render_plotTrackingFull = settings_showFullTracking.Get();
		setting_render_displayCoarseTrackingFull = settings_showCoarseTracking.Get();


	    this->settings_absVarTH = settings_absVarTH.Get();
	    this->settings_scaledVarTH = settings_scaledVarTH.Get();
	    this->settings_minRelBS = settings_minRelBS.Get();
	    this->settings_sparsity = settings_sparsity.Get();

	    // setting_desiredImmatureDensity = settings_nCandidates.Get();

	    if(settings_resetButton.Get())
	    {
	    	printf("RESET!\n");
	    	settings_resetButton.Reset();
	    	setting_fullResetRequested = true;
	    }

		// Swap frames and Process Events
		pangolin::FinishFrame();


	    if(needReset) reset_internal();
		const auto endTime = Timer::getTime();
		I3D_LOG(i3d::detail) << "Time for PangolinDSO: " << Timer::getTimeDiffMiS(startTime,endTime);
	}
	printf("QUIT Pangolin thread!\n");
	printf("I'll just kill the whole process.\nSo Long, and Thanks for All the Fish!\n");
}


void PangolinDSOViewer::close()
{
	running = false;
}

void PangolinDSOViewer::join()
{
	runThread.join();
	printf("JOINED Pangolin thread!\n");
}

void PangolinDSOViewer::reset()
{
	needReset = true;
}

void 
PangolinDSOViewer::deleteTillFrameId(size_t frameId)
{
	model3DMutex.lock();
	size_t delTillId = 0;
	for (int idx = mKeyframes.size()-1; idx >= 0; --idx)
	{
		auto& kf = mKeyframes[idx];
		if (kf->frameId >= frameId)
		{
			keyframesByKFID.erase(keyframesByKFID.find(kf->kfId));
			delTillId = idx;
		}
		else break;
	}
	//Also delete from mKeyframes[idx]
	// mKeyframes.erase(std::remove_if(mKeyframes))

	// I3D_LOG(i3d::info) << "delTillId: " << delTillId;
	// for (const auto& kf : mKeyframes)
	// {
	// 	I3D_LOG(i3d::info) << "kf before: " << kf->kfId << " frameId: " << kf->frameId << " kfInDb: " << kf->isInFernDB <<"/"<<kf->isInLoop;
	// }
	mKeyframes.erase(mKeyframes.begin()+delTillId, mKeyframes.end());
	// for (const auto& kf : mKeyframes)
	// {
	// 	I3D_LOG(i3d::info) << "kf after : " << kf->kfId << " frameId: " << kf->frameId << " kfInDb: " << kf->isInFernDB <<"/"<<kf->isInLoop;
	// }
	mMargConnections.clear();
	mEdgeConnections.clear();
	mConnections.clear();
	model3DMutex.unlock();
	

	openImagesMutex.lock();
	internalVideoImg->setBlack();
	internalKFImg->setBlack();
	internalResImg->setBlack();
	videoImgChanged= kfImgChanged= resImgChanged=true;
	openImagesMutex.unlock();
	// mAllPoses.erase(std::remove_if(mAllPoses.begin(),mAllPoses.end(),[frameId](const auto& pose){return}),
			//    mAllPoses.end());
	// std::erase(std::remove_if(mKeyframes.begin(),mKeyframes.end(),
					// [frameId](const KeyFrameDisplay* kfFrame) { return kfFrame->kfId >= keyframeId; }),
					// mKeyframes.end());
}

void PangolinDSOViewer::reset_internal()
{
	model3DMutex.lock();
	mKeyframes.clear();
	mKeyframes.clear();
	mAllPoses.clear();
	keyframesByKFID.clear();
	mConnections.clear();
	mMargConnections.clear();
	mEdgeConnections.clear();
	model3DMutex.unlock();


	openImagesMutex.lock();
	internalVideoImg->setBlack();
	internalKFImg->setBlack();
	internalResImg->setBlack();
	videoImgChanged= kfImgChanged= resImgChanged=true;
	openImagesMutex.unlock();

	needReset = false;
}


void PangolinDSOViewer::drawConstraints()
{

	auto drawSingleConnection = [](const GraphConnection& c)
							{
								Sophus::Vector3f t = c.from->camToWorld.translation().cast<float>();
								glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
								t = c.to->camToWorld.translation().cast<float>();
								glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
							};

	auto drawAllConnections = [&drawSingleConnection](const GraphConnections& connections, const Vec3f& color)
							{
								glColor3f(color[0],color[1],color[2]);
								glLineWidth(3);
								glBegin(GL_LINES);
								for (const auto& connection : connections)
								{
									if(connection.to == 0 || connection.from==0) continue;
									const int nAct = connection.bwdAct + connection.fwdAct;
									if(nAct > 0) drawSingleConnection(connection);
								}
								glEnd(); 
							};
	auto drawTrajectory = [](const KeyFrameDisplayVector& poses, const Vec3f& color)
						  {
							glColor3f(color[0],color[1],color[2]);
							glLineWidth(3);
							glBegin(GL_LINE_STRIP);
							for (const auto& frame : poses)
							{
								const Vec3f t = frame->camToWorld.translation().cast<float>();
								glVertex3f(t[0],t[1],t[2]);
							}
							glEnd();
						  };
	auto drawPoses     = [](const PoseVector& poses, const Vec3f& color)
						{
							glColor3f(color[0],color[1],color[2]);
							glLineWidth(3);
							glBegin(GL_LINE_STRIP);
							for (const auto& pose : poses)
							{
								const Vec3f t = pose.translation().cast<float>();
								glVertex3f(t[0],t[1],t[2]);
							}
							glEnd();
						};
						
	if(settings_showAllConstraints)
	{
		// draw constraints
		glLineWidth(1);
		glBegin(GL_LINES);
		glColor3f(0,1,0);
		glBegin(GL_LINES);
		for (const auto& connection : mConnections)
		{
			if(connection.to == 0 || connection.from == 0) continue;
			const int nAct = connection.bwdAct + connection.fwdAct;
			const int nMarg = connection.bwdMarg + connection.fwdMarg;
			if(nAct == 0 && nMarg > 0) drawSingleConnection(connection);
		}
		glEnd();
	}
	const Vec3f ColorBlue(0,0,1);
	const Vec3f ColorYellow(1,1,0);
	const Vec3f ColorTurqoise(0,1,1);
	const Vec3f ColorRed(1,0,0);
	const Vec3f ColorGreen(0,1,0);

	if(settings_showActiveConstraints) drawAllConnections(mConnections,ColorBlue);
	if(settings_showLoopConstraints) drawAllConnections(mEdgeConnections,ColorYellow);
	if (settings_showMargConstraints) drawAllConnections(mMargConnections,ColorTurqoise);
	if(settings_showTrajectory) drawTrajectory(mKeyframes,ColorRed);
	if(settings_showFullTrajectory) drawPoses(mAllPoses,ColorGreen);
	if (settings_showGtTrajectory) drawPoses(mGtPoses,ColorBlue);
}

void
PangolinDSOViewer::publishMargGraph(const ConnectivityMap &edgeConnectivity)
{
    if(!setting_render_display3D) return;
    if(disableAllDisplay) return;

	model3DMutex.lock();
    mMargConnections.resize(edgeConnectivity.size());
    I3D_LOG(i3d::detail) << "PangolinDSOViewer edge connections: " << mMargConnections.size();
	int runningID=0;
	int totalActFwd=0, totalActBwd=0, totalMargFwd=0, totalMargBwd=0;
    for(std::pair<uint64_t,Eigen::Vector2i> p : edgeConnectivity)
	{
		int host = (int)(p.first >> 32);
        int target = (int)(p.first & (uint64_t)0xFFFFFFFF);
		I3D_LOG(i3d::detail) << "PangolinDSOViewer Adding: " << mMargConnections[runningID].from << " " << mConnections[runningID].to << " runningId: " << runningID;
		assert(host >= 0 && target >= 0);
		if(host == target)
		{
			assert(p.second[0] == 0 && p.second[1] == 0);
			continue;
		}

		if(host > target) continue;
		auto& connection = mMargConnections[runningID];
		connection.from = keyframesByKFID.count(host) == 0 ? 0 : keyframesByKFID[host];
		connection.to = keyframesByKFID.count(target) == 0 ? 0 : keyframesByKFID[target];
		connection.fwdAct = p.second[0];
		connection.fwdMarg = p.second[1];
		totalActFwd += p.second[0];
		totalMargFwd += p.second[1];

        const uint64_t inverseKey = (((uint64_t)target) << 32) + ((uint64_t)host);
		const Vec2i st = edgeConnectivity.at(inverseKey);
		I3D_LOG(i3d::detail) << connection.from << " and " << connection.to << " st: " << st.transpose();
		connection.bwdAct = st[0];
		connection.bwdMarg = st[1];

		totalActBwd += st[0];
		totalMargBwd += st[1];

		runningID++;
	}
	model3DMutex.unlock();
}

void 
PangolinDSOViewer::publishEdgeGraph(const ConnectivityMap &edgeConnectivity)
{
    if(!setting_render_display3D) return;
    if(disableAllDisplay) return;

	model3DMutex.lock();
    mEdgeConnections.resize(edgeConnectivity.size());
    I3D_LOG(i3d::info) << "PangolinDSOViewer edge connections: " << mEdgeConnections.size();
	int runningID=0;
	int totalActFwd=0, totalActBwd=0, totalMargFwd=0, totalMargBwd=0;
    for(std::pair<uint64_t,Eigen::Vector2i> p : edgeConnectivity)
	{
		int host = (int)(p.first >> 32);
        int target = (int)(p.first & (uint64_t)0xFFFFFFFF);
		auto& connection = mEdgeConnections[runningID];
		I3D_LOG(i3d::info) << "PangolinDSOViewer Adding: " << connection.from << " " << connection.to << " runningId: " << runningID;
		assert(host >= 0 && target >= 0);
		if(host == target)
		{
			assert(p.second[0] == 0 && p.second[1] == 0);
			continue;
		}

		if(host > target) continue;
		connection.from = keyframesByKFID.count(host) == 0 ? 0 : keyframesByKFID[host];
		connection.to = keyframesByKFID.count(target) == 0 ? 0 : keyframesByKFID[target];
		connection.fwdAct = p.second[0];
		connection.fwdMarg = p.second[1];
		totalActFwd += p.second[0];
		totalMargFwd += p.second[1];

        uint64_t inverseKey = (((uint64_t)target) << 32) + ((uint64_t)host);
		Eigen::Vector2i st = edgeConnectivity.at(inverseKey);
		I3D_LOG(i3d::info) << connection.from << " and " << connection.to << " st: " << st.transpose();
		connection.bwdAct = st[0];
		connection.bwdMarg = st[1];

		totalActBwd += st[0];
		totalMargBwd += st[1];

		runningID++;
	}


	model3DMutex.unlock();
}

void 
PangolinDSOViewer::publishGraph(const ConnectivityMap &connectivity)
{
    if(!setting_render_display3D) return;
    if(disableAllDisplay) return;

	model3DMutex.lock();
    mConnections.resize(connectivity.size());
    I3D_LOG(i3d::detail) << "PangolinDSOViewer connections: " << mConnections.size();
	int runningID=0;
	int totalActFwd=0, totalActBwd=0, totalMargFwd=0, totalMargBwd=0;
    for(std::pair<uint64_t,Eigen::Vector2i> p : connectivity)
	{
		int host = (int)(p.first >> 32);
        int target = (int)(p.first & (uint64_t)0xFFFFFFFF);
		I3D_LOG(i3d::detail) << "PangolinDSOViewer Adding: " << mConnections[runningID].from << " " << mConnections[runningID].to << " runningId: " << runningID;
		assert(host >= 0 && target >= 0);
		if(host == target)
		{
			assert(p.second[0] == 0 && p.second[1] == 0);
			continue;
		}

		if(host > target) continue;
		auto& connection = mConnections[runningID];
		connection.from = keyframesByKFID.count(host) == 0 ? 0 : keyframesByKFID[host];
		connection.to = keyframesByKFID.count(target) == 0 ? 0 : keyframesByKFID[target];
		connection.fwdAct = p.second[0];
		connection.fwdMarg = p.second[1];
		totalActFwd += p.second[0];
		totalMargFwd += p.second[1];

        uint64_t inverseKey = (((uint64_t)target) << 32) + ((uint64_t)host);
		Eigen::Vector2i st = connectivity.at(inverseKey);
		I3D_LOG(i3d::detail) << mConnections[runningID].from << " and " << mConnections[runningID].to << " st: " << st.transpose();
		connection.bwdAct = st[0];
		connection.bwdMarg = st[1];

		totalActBwd += st[0];
		totalMargBwd += st[1];

		runningID++;
	}
	model3DMutex.unlock();
}

void
PangolinDSOViewer::publishCompleteTrajectory(const std::vector<std::unique_ptr<FrameHeader>>& frameHeaders)
{
	// constexpr float colorGreen[3] = {1,0,0};
	// glColor3f(colorGreen[0],colorGreen[1],colorGreen[2]);
	// glLineWidth(3);
	// glBegin(GL_LINE_STRIP);
	mAllPoses.clear();
	std::transform(frameHeaders.begin(),frameHeaders.end(),std::back_inserter(mAllPoses), [](const auto& fh)
				{
					if (fh->mFrameId > 1800)
					{
						I3D_LOG(i3d::info) << fh->mFrameId << ": " << fh->getCamToWorld().matrix3x4() << "/" << fh->getWorldPoseFromRef().matrix3x4();
					}
					return fh->getCamToWorld();
				});
	// for (const auto& fh : mAllPoses)
	// {
	// 	const Vec3f t = fh->getCamToWorld().translation().cast<float>();
	// 	glVertex3f(t[0],t[1],t[2]);
	// }
	// glEnd();
}

void PangolinDSOViewer::publishKeyframes(std::vector<FrameData *> &frames, const CameraMatrix& HCalib)
{
	if(!setting_render_display3D) return;
    if(disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lk(model3DMutex);
	// return;
    for(FrameData* fh : frames)
	{
		// I3D_LOG(i3d::info) << "publish keyframe: " << fh->mKeyFrameId;
		if(keyframesByKFID.find(fh->mKeyFrameId) == keyframesByKFID.end())
		{
			// I3D_LOG(i3d::info) << "really publish keyframe: " << fh->mKeyFrameId;
			mKeyframes.emplace_back(std::make_unique<KeyFrameDisplay>());
			// I3D_LOG(i3d::info) << "mKeyframes: " << mKeyframes.size();
			auto& kfd = mKeyframes.back();
			// I3D_LOG(i3d::info) << " mKeyframes.back()->frameId: " << mKeyframes.back()->frameId;
			// I3D_LOG(i3d::info) << " kfd->frameId: " << kfd->frameId;
			// I3D_LOG(i3d::info) << "really publish keyframe kfd: " << kfd->frameId << " fh: " << fh->mKeyFrameId;
			kfd->setFromKF(fh,HCalib);
			keyframesByKFID[fh->mKeyFrameId] = kfd.get();
			// KeyFrameDisplay* kfd = new KeyFrameDisplay();
			// kfd->setFromKF(fh, HCalib);
			// keyframesByKFID[fh->mKeyFrameId] = kfd;
			// mKeyframes.push_back(kfd);
		}
		//this else essentially safes one hashing operation
		else
		{
			keyframesByKFID[fh->mKeyFrameId]->setFromKF(fh, HCalib);
		}
	}
}

void
PangolinDSOViewer::updateKeyframePoses(std::vector<FrameHeader*> keyFrameHeaders)
{
	boost::unique_lock<boost::mutex> lk(model3DMutex);
	for (const auto& fh : keyFrameHeaders)
	{
		const auto& idx = &fh - &keyFrameHeaders[0];
		auto it = keyframesByKFID.find(idx);
		if(it != keyframesByKFID.end())
		{
			it->second->camToWorld = fh->getCamToWorld();
			
			// I3D_LOG(i3d::info) << "Pose " << fh->mFrameId << " ref: " << fh->getRefFrame()->mKeyFrameId << "/" << fh->getRefFrame()->mFrameHeader->mFrameId 
							//    << " p: " << fh->getCamToWorld().matrix3x4() << "/" << fh->getWorldPoseFromRef().matrix3x4()
							//    << " id: " << it->second->frameId << "kfid: " << it->second->kfId;
		}
	}
}

void 
PangolinDSOViewer::publishNewKeyframe(FrameData* frameData, const CameraMatrix& camMat)
{
	I3D_LOG(i3d::info) << "publishNewKeyframe!" << mKeyframes.size();
	if(!setting_render_display3D) return;
    if(disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lk(model3DMutex);
	// return;
	mKeyframes.emplace_back(std::make_unique<KeyFrameDisplay>());
	auto& kfd = mKeyframes.back();
	kfd->setFromKF(frameData, camMat);
	keyframesByKFID[frameData->mKeyFrameId] = kfd.get();

	// KeyFrameDisplay* kfd = new KeyFrameDisplay();
	// kfd->setFromKF(frameData, camMat);
	// keyframesByKFID[frameData->mKeyFrameId] = kfd;
	// mKeyframes.push_back(kfd);
	// keyframesByKFID[frameData->mKeyFrameId]->setFromKF(frameData, &camMat);
}

//void PangolinDSOViewer::publishCamPose(const FrameHeader* const frame, const CameraMatrix* const HCalib)
void PangolinDSOViewer::publishCamPose(const FrameHeader& frame, const CameraMatrix& HCalib)

{	
	I3D_LOG(i3d::info) << "publishCamPose!" << frame.mFrameId;
    if(!setting_render_display3D) return;
    if(disableAllDisplay) return;
	boost::unique_lock<boost::mutex> lk(model3DMutex);
	struct timeval time_now;
	gettimeofday(&time_now, NULL);
	lastNTrackingMs.push_back(((time_now.tv_sec-last_track.tv_sec)*1000.0f + (time_now.tv_usec-last_track.tv_usec)/1000.0f));
	if(lastNTrackingMs.size() > 10) lastNTrackingMs.pop_front();
	last_track = time_now;

	if(!setting_render_display3D) return;
	currentCam->setFromF(frame, HCalib);
    // mAllPoses.push_back(frame.getCamToWorld().translation().cast<float>());
}


void PangolinDSOViewer::pushLiveFrame(const FrameData * const image)
{
	if(!setting_render_displayVideo) return;
    if(disableAllDisplay) return;
	const auto startTime = Timer::getTime();
	uint8_t* edgePtr = reinterpret_cast<uint8_t*>(image->mFrameSet->edgeImage.ptr());
	if (image->mFrameHeader->mIsKeyFrame)
	{
		struct timeval time_now;
		gettimeofday(&time_now, NULL);
		lastNMappingMs.push_back(((time_now.tv_sec-last_map.tv_sec)*1000.0f + (time_now.tv_usec-last_map.tv_usec)/1000.0f));
		if(lastNMappingMs.size() > 10) lastNMappingMs.pop_front();
		last_map = time_now;
	}
	
	{	
		boost::unique_lock<boost::mutex> lk(openImagesMutex);
		for (int i = 0; i < w*h; ++i)
		{
			internalVideoImg->data[i] = (edgePtr[i] > 0 ? Vec3b(255,255,255) : Vec3b(0,0,0));
		}

		//We display the depth image
		if (!image->mFrameSet->depth.empty())
		{
			memcpy(internalKFImg->data, image->mFrameSet->depthJet.data, w*h*3);
			kfImgChanged = true;
		}
		videoImgChanged = true;
	}
	const auto endTime = Timer::getTime();
	I3D_LOG(i3d::info) << "Time for pushingLiveFrame: " << Timer::getTimeDiffMiS(startTime,endTime);
}

bool PangolinDSOViewer::needPushDepthImage()
{
    return setting_render_displayDepth;
}
void PangolinDSOViewer::pushDepthImage(MinimalImageB3* image)
{

    if(!setting_render_displayDepth) return;
    if(disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lk(openImagesMutex);

	struct timeval time_now;
	gettimeofday(&time_now, NULL);
	lastNMappingMs.push_back(((time_now.tv_sec-last_map.tv_sec)*1000.0f + (time_now.tv_usec-last_map.tv_usec)/1000.0f));
	if(lastNMappingMs.size() > 10) lastNMappingMs.pop_front();
	last_map = time_now;

	memcpy(internalKFImg->data, image->data, w*h*3);
	kfImgChanged=true;
}

}
}
