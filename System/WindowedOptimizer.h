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
 *  This module is very similar to DSO's local mapping.
 *  Please, consider also citing DSO:
 *   - Direct Sparse Odometry, J. Engel, V. Koltun, D. Cremers
 * 
 */

#pragma once
#include <Eigen/Eigen>
#include "../config/Defines.h"

namespace RESLAM
{
extern bool EFAdjointsValid;
extern bool EFIndicesValid;
extern bool EFDeltaValid;
template<typename Running>
class IndexThreadReduce;

class CameraMatrix;
class EFPoint;
class EFFrame;
class EdgeFrameResidual;
class EdgePixel;
class FrameData;
class AccumulatedTopHessianSSE;
class AccumulatedSCHessianSSE;
class WindowedOptimizer;
class SystemSettings;
class EFResidual
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    bool isLinearized;
    // structural pointers
    EdgeFrameResidual* data;
    int hostIDX, targetIDX;
    EFPoint* point;
    EFFrame* host;
    EFFrame* target;
    int idxInAll;
    RawResidualJacobian* J;
    VecNRf res_toZeroF;
    Vec6f JpJdF;

    // if residual is not OOB & not OUTLIER & should be used during accumulations
    bool isActiveAndIsGoodNEW;
    inline const bool &isActive() const {return isActiveAndIsGoodNEW;}
    inline EFResidual(EdgeFrameResidual* org, EFPoint* point_, EFFrame* host_, EFFrame* target_) :
        data(org), point(point_), host(host_), target(target_)
    {
        isLinearized=false;
        isActiveAndIsGoodNEW=false;
        J = new RawResidualJacobian();
        assert(((long)this)%16==0);
        assert(((long)J)%16==0);
    }
    inline ~EFResidual()
    {
        delete J;
    }
    void takeDataF();
    void fixLinearizationF(const WindowedOptimizer& ef);
};
enum EFPointStatus {PS_GOOD=0, PS_MARGINALIZE, PS_DROP};

class EFPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EFPoint(EdgePixel* d, EFFrame* host_) : data(d),host(host_)
    {
        takeData();
        stateFlag = EFPointStatus::PS_GOOD;
    }
    void takeData();

    EdgePixel* data;

    float priorF;
    float deltaF;

    // constant info (never changes in-between).
    int idxInPoints;
    EFFrame* host;

    // contains all residuals.
    std::vector<EFResidual*> residualsAll;

    float bdSumF;
    float HdiF;
    float Hdd_accLF;
    VecCf Hcd_accLF;
    float bd_accLF;
    float Hdd_accAF;
    VecCf Hcd_accAF;
    float bd_accAF;


    EFPointStatus stateFlag;
};


class EFFrame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EFFrame(FrameData* d) : data(d)
    {

        takeData();
    }
    void takeData();


    Vec6 prior;				// prior hessian (diagonal)
    Vec6 delta_prior;		// = state-state_prior (E_prior = (delta_prior)' * diag(prior) * (delta_prior)
    Vec6 delta;				// state - state_zero.



    std::vector<EFPoint*> points;
    FrameData* data;
    size_t idx;	// idx in frames.

    int frameID;
};



class WindowedOptimizer
{
//Note that all the pointers are just observers and do not manage life time!
private:
    VecX getStitchedDeltaF() const;
    float currentLambda;
    std::vector<EFPoint*> allEdges;
    std::vector<EFPoint*> allEdgesToMarg;
    std::vector<EdgeFrameResidual*> allResiduals; //pointer belongs to host frame's edge
    size_t nPoints, nFrames, nResiduals;

    std::unique_ptr<AccumulatedTopHessianSSE> accSSE_top_L;
    std::unique_ptr<AccumulatedTopHessianSSE> accSSE_top_A;
    std::unique_ptr<AccumulatedSCHessianSSE> accSSE_bot;
    size_t resInL, resInM;
    void accumulateAF_MT(MatXX &H, VecX &b);
    void accumulateLF_MT(MatXX &H, VecX &b);
    void accumulateSCF_MT(MatXX &H, VecX &b);
    IndexThreadReduce<Vec10>* red;
public:
    std::vector<EFFrame*> mEFrames;
    Mat66* adHost;
    Mat66* adTarget;
    VecC cPrior;
    VecCf cPriorF;
    Mat66f* adHostF;
    Mat66f* adTargetF;
    VecCf cDeltaF;
    Mat16f* adHTdeltaF;
    void setDeltaF(const CameraMatrix& HCalib);
    size_t resInA;
    //All connections of the local mapper that were ever in use
    ConnectivityMap mConnectivityMap;
    void setAdjointsF(const CameraMatrix& HCalib);
    auto getNbPoints() const { return nPoints; }
    auto getNbFrames() const { return nFrames; }
    double calcMEnergyF() const;
    void calcLEnergyPt(int min, int max, Vec10 *stats, int tid);
    void calcLEnergyPtSSE(int min, int max, Vec10 *stats, int tid);
    void setThreadReduce(IndexThreadReduce<Vec10>* reduce); 

    void marginalizePointsF();
    void marginalizeFrame(EFFrame* fh);

    VecX lastX;
    MatXX HM;
    VecX bM;

    std::vector<VecX> lastNullspaces_forLogging,lastNullspaces_pose,lastNullspaces_scale; //,lastNullspaces_affA,lastNullspaces_affB;
    WindowedOptimizer(const SystemSettings& settings);
    ~WindowedOptimizer();
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EFFrame* insertKeyFrame(FrameData* keyFrame, const CameraMatrix& cameraMatrix);
    EFResidual* insertResidual(EdgeFrameResidual* r);
    EFPoint* insertEdgePixel(EdgePixel* ph);
    void dropResidual(EFResidual* r);
    void removePoint(EFPoint* p);
    double calcLEnergyF_MT();   
    void solveSystemF(int iteration, double lambda, CameraMatrix& HCalib);
    void orthogonalize(VecX* b, MatXX* H);
    void resubstituteF_MT(const VecX &x, CameraMatrix& HCalib);
    void resubstituteFPt(const VecCf &xc, Mat16f* xAd, int min, int max, Vec10* stats, int tid);
    void dropPointsF();
    void makeIDX();
    std::vector<EFPoint*> allPoints;
	std::vector<EFPoint*> allPointsToMarg;
    //reset the windowed optimizer after tracking loss
    void resetWindowedOptimizer(std::vector<FrameData*>& frameHessians);
private:
    const SystemSettings& mSystemSettings;
};


}