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
#include <stdio.h>
#include <pangolin/pangolin.h>

#include "../../config/Defines.h"
#include "../MinimalImage.h"
#include "../../IOWrapper/DataStructures.h"

#include "KeyFrameDisplay.h"
#include "../../Utils/Logging.h"

namespace RESLAM
{
namespace IOWrap
{
KeyFrameDisplay::KeyFrameDisplay()
{
	mOriginalInputSparse = nullptr;
	numSparseBufferSize = 0;
	numSparsePoints = 0;

	frameId = 0;
	active = true;
    camToWorld = SE3Pose();

	needRefresh = true;

	my_scaledTH =1e10;
	my_absTH = 1e10;
	my_displayMode = 1;
	my_minRelBS = 0;
	my_sparsifyFactor = 1;

	numGLBufferPoints=0;
	bufferValid = false;
}
void KeyFrameDisplay::setFromF(const FrameHeader& frame, const CameraMatrix& HCalib)
{
    frameId = frame.mFrameId;
	kfId = frame.getRefFrame()->mKeyFrameId;
	// I3D_LOG(i3d::info) << "Adding frameId: " << frameId << " kfId: " << kfId << " timestamp: " << std::fixed << frame.mTimestamp;
	fx = HCalib.fxl();
	fy = HCalib.fyl();
	cx = HCalib.cxl();
	cy = HCalib.cyl();
    width = HCalib.width[0];//wG[0];
    height = HCalib.height[0];
	fxi = 1.0f / fx;
	fyi = 1.0f / fy;
	cxi = -cx / fx;
	cyi = -cy / fy;
    camToWorld = frame.getCamToWorld();// camToWorld;
	needRefresh = true;
}

void KeyFrameDisplay::setFromKF(const FrameData * const fh, const CameraMatrix& HCalib)
{
    setFromF(*fh->mFrameHeader, HCalib);

    const auto& edgesDetected = fh->getDetectedEdges();
    const auto& edgesGood = fh->getEdgesGood();
    const auto& edgesOut = fh->getEdgesOut();
    const auto& edgesMarginalized = fh->getEdgesMarginalized();
	constexpr bool ShowDetectedEdgePC{false};
	// add all traces, inlier and outlier points.
    const auto npoints = (ShowDetectedEdgePC ? edgesDetected.size() : 0) + edgesGood.size() + edgesMarginalized.size() + edgesOut.size();

	if(numSparseBufferSize < npoints)
	{
        if(mOriginalInputSparse != nullptr) delete mOriginalInputSparse;
		numSparseBufferSize = npoints+100;
        mOriginalInputSparse = new InputPointSparse<NUMBER_OF_VIEWER_COLORS>[numSparseBufferSize];
	}

    InputPointSparse<NUMBER_OF_VIEWER_COLORS>* pc = mOriginalInputSparse;
	numSparsePoints = 0;


	if (ShowDetectedEdgePC)
    {for(const auto& p : edgesDetected)
	{
        for (size_t clr = 0; clr < NUMBER_OF_VIEWER_COLORS; ++clr) pc[numSparsePoints].color[clr] = p->color[clr];
        pc[numSparsePoints].u = p->hostX;
        pc[numSparsePoints].v = p->hostY;
        pc[numSparsePoints].idepth = p->idepth;
		pc[numSparsePoints].idepth_hessian = 1000;
		pc[numSparsePoints].relObsBaseline = 1;
		pc[numSparsePoints].numGoodRes = 1;
		pc[numSparsePoints].status = 0;
		numSparsePoints++;
	}}

    for(const auto& p : edgesGood)
	{
        for (size_t clr = 0; clr < NUMBER_OF_VIEWER_COLORS; ++clr) pc[numSparsePoints].color[clr] = p->color[clr];
        pc[numSparsePoints].u = p->hostX;
        pc[numSparsePoints].v = p->hostY;
        pc[numSparsePoints].idepth = p->getIdepthScaled();
		pc[numSparsePoints].relObsBaseline = p->maxRelBaseline;
		pc[numSparsePoints].idepth_hessian = p->idepth_hessian;
		pc[numSparsePoints].numGoodRes =  0;
		pc[numSparsePoints].status=1;
		numSparsePoints++;
	}

    for(const auto& p : edgesMarginalized)
	{
        for (size_t clr = 0; clr < NUMBER_OF_VIEWER_COLORS; ++clr) pc[numSparsePoints].color[clr] = p->color[clr];
        pc[numSparsePoints].u = p->hostX;
        pc[numSparsePoints].v = p->hostY;
        pc[numSparsePoints].idepth = p->getIdepthScaled();
		pc[numSparsePoints].relObsBaseline = p->maxRelBaseline;
		pc[numSparsePoints].idepth_hessian = p->idepth_hessian;
		pc[numSparsePoints].numGoodRes =  0;
		pc[numSparsePoints].status=2;
		numSparsePoints++;
	}

    for(const auto& p : edgesOut)
	{
        for (size_t clr = 0; clr < NUMBER_OF_VIEWER_COLORS; ++clr) pc[numSparsePoints].color[clr] = p->color[clr];
        pc[numSparsePoints].u = p->hostX;
        pc[numSparsePoints].v = p->hostY;
        pc[numSparsePoints].idepth = p->getIdepthScaled();
		pc[numSparsePoints].relObsBaseline = p->maxRelBaseline;
		pc[numSparsePoints].idepth_hessian = p->idepth_hessian;
		pc[numSparsePoints].numGoodRes =  0;
		pc[numSparsePoints].status=3;
		numSparsePoints++;
	}
	assert(numSparsePoints <= npoints);
	camToWorld = fh->getPRE_camToWorld();
	isInFernDB = fh->isInFernDb();
	isInLoop = fh->isInLoop();
	needRefresh=true;
}

KeyFrameDisplay::~KeyFrameDisplay()
{
	if(mOriginalInputSparse != nullptr)
		delete[] mOriginalInputSparse;
}

bool KeyFrameDisplay::refreshPC(bool canRefresh, float scaledTH, float absTH, int mode, float minBS, int sparsity)
{
	if(canRefresh)
	{
		needRefresh = needRefresh ||
				my_scaledTH != scaledTH ||
				my_absTH != absTH ||
				my_displayMode != mode ||
				my_minRelBS != minBS ||
				my_sparsifyFactor != sparsity;
	}

	if(!needRefresh) return false;
	needRefresh = false;

	my_scaledTH = scaledTH;
	my_absTH = absTH;
	my_displayMode = mode;
	my_minRelBS = minBS;
	my_sparsifyFactor = sparsity;


	// if there are no vertices, done!
	if(numSparsePoints == 0)
		return false;

	// make data
    size_t patternNum = 1;
	Vec3f* tmpVertexBuffer = new Vec3f[numSparsePoints*patternNum];
	Vec3b* tmpColorBuffer = new Vec3b[numSparsePoints*patternNum];
    size_t vertexBufferNumPoints=0;

    for(size_t i=0;i<numSparsePoints;i++)
	{
		/* display modes:
		 * my_displayMode==0 - all pts, color-coded
		 * my_displayMode==1 - normal points
		 * my_displayMode==2 - active only
		 * my_displayMode==3 - nothing
		 */

		if(my_displayMode==1 && mOriginalInputSparse[i].status != 1 && mOriginalInputSparse[i].status!= 2) continue;
		if(my_displayMode==2 && mOriginalInputSparse[i].status != 1) continue;
		if(my_displayMode>2) continue;

        if(mOriginalInputSparse[i].idepth < 0) continue;

        const float depth = 1.0f / mOriginalInputSparse[i].idepth;
		float depth4 = depth*depth; 
		depth4 *= depth4;
		const float var = (1.0f / (mOriginalInputSparse[i].idepth_hessian+0.01));

		if(var * depth4 > my_scaledTH)
			continue;

		if(var > my_absTH)
			continue;

		if(mOriginalInputSparse[i].relObsBaseline < my_minRelBS)
			continue;

        for(size_t pnt=0;pnt<patternNum;pnt++)
		{

			if(my_sparsifyFactor > 1 && rand()%my_sparsifyFactor != 0) continue;

            int dx = 0, dy = 0;
			tmpVertexBuffer[vertexBufferNumPoints][0] = ((mOriginalInputSparse[i].u+dx)*fxi + cxi) * depth;
			tmpVertexBuffer[vertexBufferNumPoints][1] = ((mOriginalInputSparse[i].v+dy)*fyi + cyi) * depth;
			tmpVertexBuffer[vertexBufferNumPoints][2] = depth*(1 + 2*fxi * (rand()/(float)RAND_MAX-0.5f));

			if(my_displayMode==0)
			{
				if(mOriginalInputSparse[i].status==0)
				{
					tmpColorBuffer[vertexBufferNumPoints][0] = 0;
					tmpColorBuffer[vertexBufferNumPoints][1] = 255;
					tmpColorBuffer[vertexBufferNumPoints][2] = 255;
				}
				else if(mOriginalInputSparse[i].status==1)
				{
					tmpColorBuffer[vertexBufferNumPoints][0] = 0;
					tmpColorBuffer[vertexBufferNumPoints][1] = 255;
					tmpColorBuffer[vertexBufferNumPoints][2] = 0;
				}
				else if(mOriginalInputSparse[i].status==2)
				{
					tmpColorBuffer[vertexBufferNumPoints][0] = 0;
					tmpColorBuffer[vertexBufferNumPoints][1] = 0;
					tmpColorBuffer[vertexBufferNumPoints][2] = 255;
				}
				else if(mOriginalInputSparse[i].status==3)
				{
					tmpColorBuffer[vertexBufferNumPoints][0] = 255;
					tmpColorBuffer[vertexBufferNumPoints][1] = 0;
					tmpColorBuffer[vertexBufferNumPoints][2] = 0;
				}
				else
				{
					tmpColorBuffer[vertexBufferNumPoints][0] = 255;
					tmpColorBuffer[vertexBufferNumPoints][1] = 255;
					tmpColorBuffer[vertexBufferNumPoints][2] = 255;
				}

			}
			else
			{
                tmpColorBuffer[vertexBufferNumPoints][0] = mOriginalInputSparse[i].color[2];
                tmpColorBuffer[vertexBufferNumPoints][1] = mOriginalInputSparse[i].color[1];
                tmpColorBuffer[vertexBufferNumPoints][2] = mOriginalInputSparse[i].color[0];
			}
			vertexBufferNumPoints++;


			assert(vertexBufferNumPoints <= numSparsePoints*patternNum);
		}
	}

	if(vertexBufferNumPoints==0)
	{
		delete[] tmpColorBuffer;
		delete[] tmpVertexBuffer;
		return true;
	}

	numGLBufferGoodPoints = vertexBufferNumPoints;
	if(numGLBufferGoodPoints > numGLBufferPoints)
	{
		numGLBufferPoints = vertexBufferNumPoints*1.3;
		vertexBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_FLOAT, 3, GL_DYNAMIC_DRAW );
		colorBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW );
	}
	vertexBuffer.Upload(tmpVertexBuffer, sizeof(float)*3*numGLBufferGoodPoints, 0);
	colorBuffer.Upload(tmpColorBuffer, sizeof(unsigned char)*3*numGLBufferGoodPoints, 0);
	bufferValid=true;
	delete[] tmpColorBuffer;
	delete[] tmpVertexBuffer;


	return true;
}

void KeyFrameDisplay::drawCamKF(float lineWidth, float* color, float sizeFactor, const size_t idx)
{
	if(!isInFernDB && !isInLoop) 
		return;
	drawCam(lineWidth,color,sizeFactor,idx);
}

void KeyFrameDisplay::drawCam(float lineWidth, float* color, float sizeFactor, const size_t idx)
{
	if(width == 0)
		return;


	float sz=sizeFactor;

	glPushMatrix();

		Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
		glMultMatrixf((GLfloat*)m.data());
		// I3D_LOG(i3d::info) <<"drawCam camToWorld.matrix(): " << camToWorld.matrix();
		if(color == 0)
		{
			glColor3f(1,0,0);
		}
		else
			glColor3f(color[0],color[1],color[2]);

		if (isInFernDB || isInLoop) glColor3f(0.0,0.0,1.0);


		glLineWidth(lineWidth);
		glBegin(GL_LINES);
		glVertex3f(0,0,0);
		glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz);
		glVertex3f(0,0,0);
		glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz);
		glVertex3f(0,0,0);
		glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz);
		glVertex3f(0,0,0);
		glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz);

		glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz);
		glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz);

		glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz);
		glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz);

		glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz);
		glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz);

		glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz);
		glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz);
		glEnd();
		pangolin::GlFont::I().Text("f "+std::to_string(frameId)+"/ kf "+std::to_string(kfId)).Draw(0,0,0);

	glPopMatrix();
}

void KeyFrameDisplay::drawPC(float pointSize)
{
	if(!bufferValid || numGLBufferGoodPoints == 0)
		return;

	glDisable(GL_LIGHTING);
	glPushMatrix();
	Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
	glMultMatrixf((GLfloat*)m.data());
	glPointSize(pointSize);

	colorBuffer.Bind();
	glColorPointer(colorBuffer.count_per_element, colorBuffer.datatype, 0, 0);
	glEnableClientState(GL_COLOR_ARRAY);

	vertexBuffer.Bind();
	glVertexPointer(vertexBuffer.count_per_element, vertexBuffer.datatype, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glDrawArrays(GL_POINTS, 0, numGLBufferGoodPoints);
	glDisableClientState(GL_VERTEX_ARRAY);
	vertexBuffer.Unbind();

	glDisableClientState(GL_COLOR_ARRAY);
	colorBuffer.Unbind();

	glPopMatrix();
}

}
}
