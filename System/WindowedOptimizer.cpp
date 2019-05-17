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

#include "WindowedOptimizer.h"
#include "../IOWrapper/DataStructures.h"
#include "../Utils/AccumulatedTopHessian.h"
#include "../Utils/AccumulatedSCHessian.h"
#include "../System/SystemSettings.h"
namespace RESLAM
{
    
bool EFAdjointsValid = false;
bool EFIndicesValid = false;
bool EFDeltaValid = false;

void 
WindowedOptimizer::setThreadReduce(IndexThreadReduce<Vec10>* reduce) 
{ 
    red = reduce;
}

WindowedOptimizer::WindowedOptimizer(const SystemSettings& settings): mSystemSettings(settings)
{
    nFrames = nResiduals = nPoints = 0;
    adHost = nullptr;
    adTarget = nullptr;
    red  = nullptr;
    adHostF = nullptr;
    adTargetF = nullptr;
    adHTdeltaF = nullptr;
    accSSE_top_L = std::make_unique<AccumulatedTopHessianSSE>();
    accSSE_top_A = std::make_unique<AccumulatedTopHessianSSE>();
    accSSE_bot = std::make_unique<AccumulatedSCHessianSSE>();

    resInA = resInL = resInM = 0;
    currentLambda = 0;
    HM = MatXX::Zero(CPARS,CPARS);
    bM = VecX::Zero(CPARS);

}

WindowedOptimizer::~WindowedOptimizer()
{
    for(auto* f : mEFrames)
    {
        for(auto* p : f->points)
        {
            for(auto* r : p->residualsAll)
            {
                r->data->efResidual = nullptr;
                delete r;
            }
            p->data->efPoint = nullptr;
            delete p;
        }
        f->data->efFrame = nullptr;
        delete f;
    }

    if(adHost != 0) delete[] adHost;
    if(adTarget != 0) delete[] adTarget;
    if(adHostF != 0) delete[] adHostF;
    if(adTargetF != 0) delete[] adTargetF;
    if(adHTdeltaF != 0) delete[] adHTdeltaF;
}

VecX 
WindowedOptimizer::getStitchedDeltaF() const
{
    VecX d = VecX(CPARS+nFrames*HessianSize); d.head<CPARS>() = cDeltaF.cast<double>();
    for(size_t h=0;h<nFrames;h++) d.segment<HessianSize>(CPARS+HessianSize*h) = mEFrames[h]->delta;
    return d;
}

//this is eq. 19 from the DSO paper! marginalized energy
double 
WindowedOptimizer::calcMEnergyF() const
{

    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    VecX delta = getStitchedDeltaF();
    return delta.dot(2*bM + HM*delta); //this is eq. 19 from the paper! marginalized energy
}

void 
WindowedOptimizer::marginalizePointsF()
{
    I3D_LOG(i3d::debug) << "marginalizePointsF!";
    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);


    allEdgesToMarg.clear();
    for(EFFrame* f : mEFrames)
    {
        for(int i=0;i<(int)f->points.size();i++)
        {
            EFPoint* p = f->points[i];
            if(p->stateFlag == EFPointStatus::PS_MARGINALIZE)
            {
                p->priorF *= setting_idepthFixPriorMargFac;
                for(EFResidual* r : p->residualsAll)
                    if(r->isActive())
                        mConnectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][1]++;
                allEdgesToMarg.push_back(p);
            }
        }
    }
    I3D_LOG(i3d::debug) << "Marginalizing: " << allEdgesToMarg.size();
    accSSE_bot->setZero(nFrames);
    accSSE_top_A->setZero(nFrames);
    for(EFPoint* p : allEdgesToMarg)
    {
        accSSE_top_A->addPoint<2>(p,this);
        accSSE_bot->addPoint(p,false);
        removePoint(p);
    }
    MatXX M, Msc;
    VecX Mb, Mbsc;
    accSSE_top_A->stitchDouble(M,Mb,this,false,false);
    accSSE_bot->stitchDouble(Msc,Mbsc,this);

    resInM+= accSSE_top_A->nres[0];

    MatXX H =  M-Msc;
    VecX b =  Mb-Mbsc;

    if(setting_solverMode & SOLVER_ORTHOGONALIZE_POINTMARG)
    {
        // have a look if prior is there.
        bool haveFirstFrame = false;
        for(EFFrame* f : mEFrames) if(f->frameID==0) haveFirstFrame=true;

        if(!haveFirstFrame)
            orthogonalize(&b, &H);

    }

    HM += setting_margWeightFac*H;
    bM += setting_margWeightFac*b;

    if(setting_solverMode & SOLVER_ORTHOGONALIZE_FULL)
        orthogonalize(&bM, &HM);

    EFIndicesValid = false;
    makeIDX();
}


void 
WindowedOptimizer::setAdjointsF(const CameraMatrix& HCalib)
{

    if(adHost != 0) delete[] adHost;
    if(adTarget != 0) delete[] adTarget;
    adHost = new HessianType[nFrames*nFrames];
    adTarget = new HessianType[nFrames*nFrames];

    for(size_t h = 0;h < nFrames; h++)
        for(size_t t = 0; t < nFrames; t++)
        {
            FrameData* host = mEFrames[h]->data;
            FrameData* target = mEFrames[t]->data;

            SE3Pose hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();


            HessianType AH = HessianType::Identity();
            AH.topLeftCorner<6,6>() = -hostToTarget.Adj().transpose();/*Mat88::Identity();*/
            HessianType AT = HessianType::Identity();


            AH.block<3,MOTION_DOF>(0,0) *= SCALE_XI_TRANS;
            AH.block<3,MOTION_DOF>(3,0) *= SCALE_XI_ROT;

            AT.block<3,MOTION_DOF>(0,0) *= SCALE_XI_TRANS;
            AT.block<3,MOTION_DOF>(3,0) *= SCALE_XI_ROT;


            adHost[h+t*nFrames] = AH;
            adTarget[h+t*nFrames] = AT;
        }
    cPrior = VecC::Constant(setting_initialCameraMatrix);


    if(adHostF != 0) delete[] adHostF;
    if(adTargetF != 0) delete[] adTargetF;
    adHostF = new HessianTypef[nFrames*nFrames];
    adTargetF = new HessianTypef[nFrames*nFrames];

    for(size_t h=0;h<nFrames;h++)
        for(size_t t=0;t<nFrames;t++)
        {
            adHostF[h+t*nFrames] = adHost[h+t*nFrames].cast<float>();
            adTargetF[h+t*nFrames] = adTarget[h+t*nFrames].cast<float>();
        }

    cPriorF = cPrior.cast<float>();


    EFAdjointsValid = true;
}

void 
EFFrame::takeData()
{
    prior = data->getPrior().head<HessianSize>();
    delta = data->get_state_minus_stateZero().head<HessianSize>();
    delta_prior =  (data->get_state() - data->getPriorZero()).head<HessianSize>();
    frameID = data->mKeyFrameId;
    I3D_LOG(i3d::info) << "takeData: " << frameID << " = " << data->mKeyFrameId;
}

EFFrame* 
WindowedOptimizer::insertKeyFrame(FrameData *keyFrame, const CameraMatrix& cameraMatrix)
{
    EFFrame* eff = new EFFrame(keyFrame);
    eff->idx = mEFrames.size();
    mEFrames.push_back(eff);

    nFrames++;
    keyFrame->efFrame = eff;

    assert(HM.cols() == HessianSize*nFrames+CPARS-HessianSize);
    bM.conservativeResize(HessianSize*nFrames+CPARS);
    HM.conservativeResize(HessianSize*nFrames+CPARS,HessianSize*nFrames+CPARS);
    bM.tail<HessianSize>().setZero();
    HM.rightCols<HessianSize>().setZero();
    HM.bottomRows<HessianSize>().setZero();

    EFIndicesValid = false;
    EFAdjointsValid = false;
    EFDeltaValid = false;
    
    setAdjointsF(cameraMatrix);
    makeIDX();


    for(EFFrame* fh2 : mEFrames)
    {
        I3D_LOG(i3d::detail) << "eff->frameID: " << eff->frameID << " fh2->frameID: " << fh2->frameID;
        mConnectivityMap[(((uint64_t)eff->frameID) << 32) + ((uint64_t)fh2->frameID)] = Eigen::Vector2i(0,0);
        if(fh2 != eff)
            mConnectivityMap[(((uint64_t)fh2->frameID) << 32) + ((uint64_t)eff->frameID)] = Eigen::Vector2i(0,0);
    }

    return eff;
}

EFResidual* 
WindowedOptimizer::insertResidual(EdgeFrameResidual* r)
{
    I3D_LOG(i3d::debug) << "insertResidual: " << r->host->mKeyFrameId << " instanceCounter: " << r->instanceCounter 
                       << " efPoint: " << r->edgePixel->efPoint << " EdgeFrameResidual r: " << r << " r->edgePixel: " << r->edgePixel
                       << " from host: " << r->edgePixel->host->mKeyFrameId;
    EFResidual* efr = new EFResidual(r, r->edgePixel->efPoint, r->host->efFrame, r->target->efFrame);
    efr->idxInAll = r->edgePixel->efPoint->residualsAll.size();
    r->edgePixel->efPoint->residualsAll.push_back(efr);

    I3D_LOG(i3d::debug) << "Adding: " << (((uint64_t)efr->host->frameID) << 32) + ((uint64_t)efr->target->frameID)
                       << " frameID host: " << (((uint64_t)efr->host->frameID) << 32) << " " << efr->host->frameID
                       << " frameID target: " << ((uint64_t)efr->target->frameID) << " " << efr->target->frameID;
    mConnectivityMap[(((uint64_t)efr->host->frameID) << 32) + ((uint64_t)efr->target->frameID)][0]++;

    nResiduals++;
    r->efResidual = efr;
    return efr;
}


EFPoint* 
WindowedOptimizer::insertEdgePixel(EdgePixel* ph)
{
    EFPoint* efp = new EFPoint(ph, ph->host->efFrame);
    efp->idxInPoints = ph->host->efFrame->points.size();
    ph->host->efFrame->points.push_back(efp);
    I3D_LOG(i3d::debug) << "Adding point to EFFrame: " << ph->host->efFrame->points.size()
                       << " EFFrame->efPoint: " << efp << " ph: " << ph << " " << efp->idxInPoints;
    nPoints++;
    ph->efPoint = efp;

    EFIndicesValid = false;

    return efp;
}
void 
WindowedOptimizer::dropResidual(EFResidual* r)
{
    EFPoint* p = r->point;
    assert(r == p->residualsAll[r->idxInAll]);

    p->residualsAll[r->idxInAll] = p->residualsAll.back();
    p->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
    p->residualsAll.pop_back();

    mConnectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][0]--;
    nResiduals--;
    r->data->efResidual=nullptr;
    delete r;
}
void 
WindowedOptimizer::setDeltaF(const CameraMatrix& HCalib)
{
    if(adHTdeltaF != 0) delete[] adHTdeltaF;
    adHTdeltaF = new Mat16f[nFrames*nFrames];
    for(size_t h=0;h<nFrames;h++)
        for(size_t t=0;t<nFrames;t++)
        {
            size_t idx = h+t*nFrames;
            adHTdeltaF[idx] = mEFrames[h]->data->get_state_minus_stateZero().head<HessianSize>().cast<float>().transpose() * adHostF[idx]
                             +mEFrames[t]->data->get_state_minus_stateZero().head<HessianSize>().cast<float>().transpose() * adTargetF[idx];
        }

    cDeltaF = HCalib.value_minus_value_zero.cast<float>();
    for (auto* f : mEFrames)
    {
        f->delta = f->data->get_state_minus_stateZero().head<HessianSize>();
        f->delta_prior = (f->data->get_state() - f->data->getPriorZero()).head<HessianSize>();
        I3D_LOG(i3d::debug) << "EFFrame! with point size " << f->points.size();
        for(EFPoint* p : f->points)
        {

            p->deltaF = p->data->getIdepth() - p->data->getIdepthZero();
            if (std::isnan(p->data->getIdepth()) || std::isnan(p->data->getIdepthZero()))
            {
                I3D_LOG(i3d::debug) << "Problem!! idepth or idepth_zero == 0";
                exit(0);
            }
        }
    }
    EFDeltaValid = true;
}

void 
WindowedOptimizer::removePoint(EFPoint* p)
{
    I3D_LOG(i3d::debug) << "removePoint: " << p << " at: " << p->idxInPoints << " host: " << p->host->idx;
    for(EFResidual* r : p->residualsAll)
        dropResidual(r);

    EFFrame* h = p->host;
    h->points[p->idxInPoints] = h->points.back();
    h->points[p->idxInPoints]->idxInPoints = p->idxInPoints;
    h->points.pop_back();

    nPoints--;
    p->data->efPoint = nullptr;

    EFIndicesValid = false;

    delete p;
}


void 
WindowedOptimizer::calcLEnergyPt(int min, int max, Vec10* stats, int tid)
{
    VecCf dc = cDeltaF;
    double E{0};
    for(int i = min; i < max; ++i)
    {
        EFPoint* p = allEdges[i];
        float dd = p->deltaF;

        for(EFResidual* r : p->residualsAll)
        {
            if(!r->isLinearized || !r->isActive()) continue;

            const Mat16f& dp = adHTdeltaF[r->hostIDX+nFrames*r->targetIDX];
            RawResidualJacobian* rJ = r->J;
            // compute Jp*delta
            const float Jp_delta_x_1 =  rJ->Jpdxi[0].dot(dp.head<6>())+rJ->Jpdc[0].dot(dc)+rJ->Jpdd[0]*dd;
            const float Jp_delta_y_1 =  rJ->Jpdxi[1].dot(dp.head<6>())+rJ->Jpdc[1].dot(dc)+rJ->Jpdd[1]*dd;

            //no pattern, thus we stay with float!
            const float Jdelta = rJ->JIdx[0]*Jp_delta_x_1+rJ->JIdx[1]*Jp_delta_y_1;
            const float r0 = r->res_toZeroF;
            E += (2*r0+Jdelta)*Jdelta;
        }
        E  += (p->deltaF*p->deltaF*p->priorF);
    }
    (*stats)[0] = E;
}


double 
WindowedOptimizer::calcLEnergyF_MT()
{
    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    double E = 0;
    for(EFFrame* f : mEFrames)
        E += f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior);

    E += cDeltaF.cwiseProduct(cPriorF).dot(cDeltaF);
    I3D_LOG(i3d::nothing) << "calcLEnergyF_MT!! red: " << red << " stats: " << red->stats.transpose();
    if (mSystemSettings.SystemMultiThreading)// && false)
    {
        red->reduce(boost::bind(&WindowedOptimizer::calcLEnergyPt, this, _1, _2, _3, _4), 
                    0, allPoints.size(), 50);
        return E+red->stats[0];
    }
    else
    {
        Vec10 stats;
        stats.setZero();
        calcLEnergyPt(0,allEdges.size(),&stats,0);
        return E+stats[0];
    }
}

void 
EFPoint::takeData()
{
    const float setting_idepthFixPrior = 50*50;
    priorF = data->hasDepthPrior ? setting_idepthFixPrior*SCALE_IDEPTH*SCALE_IDEPTH : 0;
    if(setting_solverMode & SOLVER_REMOVE_POSEPRIOR) priorF=0;

    deltaF = data->getIdepth()-data->getIdepthZero();

    I3D_LOG(i3d::debug) << "EFPoint::takeData() : at ("<< data->hostX << ", " << data->hostY << ")" << data->getIdepth()<<"-"<<data->getIdepthZero();
    assert(!std::isnan(deltaF));
}

void 
EFResidual::fixLinearizationF(const WindowedOptimizer& ef)
{
    const Vec6f dp = ef.adHTdeltaF[hostIDX+ef.getNbFrames()*targetIDX];


    const Vec2f Jp_delta(J->Jpdxi[0].dot(dp.head<6>())+J->Jpdc[0].dot(ef.cDeltaF)+J->Jpdd[0]*point->deltaF,
                         J->Jpdxi[1].dot(dp.head<6>())+J->Jpdc[1].dot(ef.cDeltaF)+J->Jpdd[1]*point->deltaF);

    //We don't have a pattern
    res_toZeroF = J->resF-J->JIdx.dot(Jp_delta); //should be float now

    isLinearized = true;
}

void 
EFResidual::takeDataF()
{
    std::swap<RawResidualJacobian*>(J, data->J);

    Vec2f JI_JI_Jd = J->JIdx2 * J->Jpdd;

    for(int i=0;i<6;i++)
        JpJdF[i] = J->Jpdxi[0][i]*JI_JI_Jd[0] + J->Jpdxi[1][i] * JI_JI_Jd[1];

}

void 
WindowedOptimizer::makeIDX()
{
    for(size_t idx = 0; idx < mEFrames.size(); idx++)
        mEFrames[idx]->idx = idx;


    allEdges.clear();

    for(EFFrame* f : mEFrames)
        for(EFPoint* p : f->points)
        {
            allEdges.push_back(p);
            for(EFResidual* r : p->residualsAll)
            {
                r->hostIDX = r->host->idx;
                r->targetIDX = r->target->idx;
            }
        }


    EFIndicesValid=true;
}

void 
WindowedOptimizer::dropPointsF()
{
    auto nRemoved = 0;
    for(EFFrame* f : mEFrames)
    {
        for(int i=0;i<(int)f->points.size();i++)
        {
            EFPoint* p = f->points[i];
            if(p->stateFlag == EFPointStatus::PS_DROP)
            {
                I3D_LOG(i3d::debug) << "Removing " << p << " because of PS_DROP!";
                removePoint(p);
                i--;
                nRemoved++;
            }
        }
    }
    I3D_LOG(i3d::debug) << "nRemoved points: " << nRemoved;
    EFIndicesValid = false;
    makeIDX();
}

void 
WindowedOptimizer::marginalizeFrame(EFFrame* fh)
{

    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    assert((int)fh->points.size()==0);
    const int ndim = nFrames*HessianSize+CPARS-HessianSize;// new dimension
    const int odim = nFrames*HessianSize+CPARS;// old dimension


    if((int)fh->idx != (int)mEFrames.size()-1)
    {
        int io = fh->idx*HessianSize+CPARS;	// index of frame to move to end
        int ntail = HessianSize*(nFrames-fh->idx-1);
        assert((io+HessianSize+ntail) == nFrames*HessianSize+CPARS);

        Vec6 bTmp = bM.segment<HessianSize>(io);
        VecX tailTMP = bM.tail(ntail);
        bM.segment(io,ntail) = tailTMP;
        bM.tail<HessianSize>() = bTmp;

        MatXX HtmpCol = HM.block(0,io,odim,HessianSize);
        MatXX rightColsTmp = HM.rightCols(ntail);
        HM.block(0,io,odim,ntail) = rightColsTmp;
        HM.rightCols(HessianSize) = HtmpCol;

        MatXX HtmpRow = HM.block(io,0,HessianSize,odim);
        MatXX botRowsTmp = HM.bottomRows(ntail);
        HM.block(io,0,ntail,odim) = botRowsTmp;
        HM.bottomRows(HessianSize) = HtmpRow;
    }


//	// marginalize. First add prior here, instead of to active.
    HM.bottomRightCorner<HessianSize,HessianSize>().diagonal() += fh->prior;
    bM.tail<HessianSize>() += fh->prior.cwiseProduct(fh->delta_prior);

    VecX SVec = (HM.diagonal().cwiseAbs()+VecX::Constant(HM.cols(), 10)).cwiseSqrt();
    VecX SVecI = SVec.cwiseInverse();

    // scale!
    MatXX HMScaled = SVecI.asDiagonal() * HM * SVecI.asDiagonal();
    VecX bMScaled =  SVecI.asDiagonal() * bM;

    // invert bottom part!
    HessianType hpi = HMScaled.bottomRightCorner<HessianSize,HessianSize>(); //H_bb
    hpi = 0.5f*(hpi+hpi);
    hpi = hpi.inverse();
    hpi = 0.5f*(hpi+hpi); //hpi = inv(H_bb)

    // schur-complement!

    MatXX bli = HMScaled.bottomLeftCorner(HessianSize,ndim).transpose() * hpi;  //H_ba'* inv(H_bb) = H_ab * inv(H_bb)
    //hat(H_aa) = H_aa - H_ab * inv(H_bb)* H_ba
    HMScaled.topLeftCorner(ndim,ndim).noalias() -= bli * HMScaled.bottomLeftCorner(HessianSize,ndim); //this is equation 17

    //hat b_a = b_a - H_ab * inv(H_bb) * b_b (I'm unsure about the b_b)
    bMScaled.head(ndim).noalias() -= bli*bMScaled.tail<HessianSize>();//this is equation 18

    //unscale!
    HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
    bMScaled = SVec.asDiagonal() * bMScaled;

    // set.
    HM = 0.5*(HMScaled.topLeftCorner(ndim,ndim) + HMScaled.topLeftCorner(ndim,ndim).transpose());
    bM = bMScaled.head(ndim);

    // remove from vector, without changing the order!
    for(unsigned int i=fh->idx; i+1<mEFrames.size();i++)
    {
        mEFrames[i] = mEFrames[i+1];
        mEFrames[i]->idx = i;
    }
    mEFrames.pop_back();
    nFrames--;
    fh->data->efFrame=nullptr;

    assert((int)mEFrames.size()*HessianSize+CPARS == (int)HM.rows());
    assert((int)mEFrames.size()*HessianSize+CPARS == (int)HM.cols());
    assert((int)mEFrames.size()*HessianSize+CPARS == (int)bM.size());
    assert((int)mEFrames.size() == (int)nFrames);
    EFIndicesValid = false;
    EFAdjointsValid = false;
    EFDeltaValid = false;

    makeIDX();
    delete fh;
}

void
WindowedOptimizer::resetWindowedOptimizer(std::vector<FrameData*>& frameHessians)
{
    //drop frames
    for (EFFrame* efFrame : mEFrames)
    {       
        //drop their points
        for (EFPoint* p : efFrame->points)
        {
            //drop all the residuals
            std::for_each(p->residualsAll.begin(),p->residualsAll.end(),[](EFResidual* r){ delete r; });
            delete p;
            p = nullptr;
        }
        delete efFrame;
        efFrame = nullptr;
    }
    mEFrames.clear();
    for (auto& fh : frameHessians)
            fh->efFrame = nullptr;
    
    //drop all the observations
    for (auto& fh: frameHessians)
    {
        fh->resetFrame();
    }
    //Reset connections
    mConnectivityMap.clear();
    lastX.setZero();
    HM.setZero();
    bM.setZero();

    nResiduals = 0;
    nFrames = 0;
}

void 
WindowedOptimizer::solveSystemF(int iteration, double lambda, CameraMatrix& HCalib)
{
    if(setting_solverMode & SOLVER_USE_GN) lambda = 0;
    if(setting_solverMode & SOLVER_FIX_LAMBDA) lambda = 1e-5;
    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);
    MatXX HL_top, HA_top, H_sc;
    VecX  bL_top, bA_top, bM_top, b_sc;
    accumulateAF_MT(HA_top, bA_top); //CHECKED
    accumulateLF_MT(HL_top, bL_top);
    accumulateSCF_MT(H_sc, b_sc);
    if (!mSystemSettings.LocalMapperDoMarginalize)
    {
        bM.setZero();
        HM.setZero();
    }
    bM_top = (bM+ HM * getStitchedDeltaF());
    MatXX HFinal_top;
    VecX bFinal_top;
    I3D_LOG(i3d::detail) << " HA_top: " << HA_top << "HL_top: " << HL_top << " H_sc: " << H_sc;
    I3D_LOG(i3d::detail) << " bA_top: " << bA_top << "bL_top: " << bL_top << " b_sc: " << b_sc;
    if(setting_solverMode & SOLVER_ORTHOGONALIZE_SYSTEM)
    {
        // have a look if prior is there.
        bool haveFirstFrame = false;
        for(EFFrame* f : mEFrames) if (f->frameID == 0) haveFirstFrame = true;
        MatXX HT_act =  HL_top + HA_top - H_sc;
        VecX bT_act =   bL_top + bA_top - b_sc;
        if(!haveFirstFrame)
            orthogonalize(&bT_act, &HT_act);
        HFinal_top = HT_act + HM;
        bFinal_top = bT_act + bM_top;

        for(size_t i = 0; i < HessianSize*nFrames+CPARS; i++) HFinal_top(i,i) *= (1+lambda);
    }
    else
    {
        HFinal_top = HL_top + HM + HA_top;
        bFinal_top = bL_top + bM_top + bA_top - b_sc;
        for(size_t i = 0; i < HessianSize*nFrames+CPARS; i++) HFinal_top(i,i) *= (1+lambda);
        HFinal_top -= H_sc * (1.0f/(1+lambda));
    }

    VecX x;
    if(setting_solverMode & SOLVER_SVD)
    {
        VecX SVecI = HFinal_top.diagonal().cwiseSqrt().cwiseInverse();
        MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
        VecX bFinalScaled  = SVecI.asDiagonal() * bFinal_top;
        Eigen::JacobiSVD<MatXX> svd(HFinalScaled, Eigen::ComputeThinU | Eigen::ComputeThinV);

        VecX S = svd.singularValues();
        double minSv = 1e10, maxSv = 0;
        for(int i=0;i<S.size();i++)
        {
            if(S[i] < minSv) minSv = S[i];
            if(S[i] > maxSv) maxSv = S[i];
        }

        VecX Ub = svd.matrixU().transpose()*bFinalScaled;
        int setZero=0;
        for(int i=0;i<Ub.size();i++)
        {
            if(S[i] < setting_solverModeDelta*maxSv)
            { Ub[i] = 0; setZero++; }

            if((setting_solverMode & SOLVER_SVD_CUT7) && (i >= Ub.size()-7))
            { Ub[i] = 0; setZero++; }

            else Ub[i] /= S[i];
        }
        x = SVecI.asDiagonal() * svd.matrixV() * Ub;

    }
    else
    {

        VecX SVecI = (HFinal_top.diagonal()+VecX::Constant(HFinal_top.cols(), 10)).cwiseSqrt().cwiseInverse();
        MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
        x = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_top);//  SVec.asDiagonal() * svd.matrixV() * Ub;
    }

    if((setting_solverMode & SOLVER_ORTHOGONALIZE_X) || (iteration >= 2 && (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER)))
    {
        VecX xOld = x;
        orthogonalize(&x, 0);
    }
    /** Note for x vector:
     * First 4 elements are camera parameters and then we have CPARS + HessianSize*idx 6 DOF parameters
     * basically we start at pos 4, 10, 16 for the camera parameters.
      */
    if (!std::isfinite(x[0]))
    {
        I3D_LOG(i3d::debug) << "STEP NOT FINITE! x: " << x.transpose();
    }

    lastX = x;
    currentLambda = lambda;
    resubstituteF_MT(x, HCalib);
    currentLambda=0;
}

void 
WindowedOptimizer::resubstituteF_MT(const VecX& x, CameraMatrix& HCalib)
{
    assert(x.size() == CPARS+nFrames*HessianSize);

    VecXf xF = x.cast<float>();
    HCalib.step = - x.head<CPARS>();
    if (!std::isfinite(x[0]))
    {
        I3D_LOG(i3d::debug) << "STEP NOT FINITE! x: " << x.transpose();
    }
    Mat16f* xAd = new Mat16f[nFrames*nFrames];
    //TODO: Does not change result, remove setZero!
    for (size_t i = 0; i < nFrames*nFrames; ++i)
    {
        xAd[i].setZero();
    }
    VecCf cstep = xF.head<CPARS>();
    //h->idx = 0, 1,...., thus we take 6 elements starting at 4, 10,...
    for(EFFrame* h : mEFrames)
    {
        h->data->step.head<HessianSize>() = - x.segment<HessianSize>(CPARS+HessianSize*h->idx);
        for(EFFrame* t : mEFrames)
            xAd[nFrames*h->idx + t->idx] = xF.segment<HessianSize>(CPARS+HessianSize*h->idx).transpose() *   adHostF[h->idx+nFrames*t->idx]
                        + xF.segment<HessianSize>(CPARS+HessianSize*t->idx).transpose() * adTargetF[h->idx+nFrames*t->idx];
    }

    if(mSystemSettings.SystemMultiThreading)// && false)
    {
        red->reduce(boost::bind(&WindowedOptimizer::resubstituteFPt,
                    this, cstep, xAd,  _1, _2, _3, _4), 0, allEdges.size(), 50);

    }
    else
    {
        resubstituteFPt(cstep, xAd, 0, allEdges.size(), nullptr, 0);
    }
    delete[] xAd;
}
void WindowedOptimizer::orthogonalize(VecX* b, MatXX* H)
{
    // decide to which nullspaces to orthogonalize.
    std::vector<VecX> ns;
    ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());
    ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end());

    // make Nullspaces matrix
    MatXX N(ns[0].rows(), ns.size());
    for(unsigned int i=0;i<ns.size();i++)
        N.col(i) = ns[i].normalized();

    // compute Npi := N * (N' * N)^-1 = pseudo inverse of N.
    Eigen::JacobiSVD<MatXX> svdNN(N, Eigen::ComputeThinU | Eigen::ComputeThinV);

    VecX SNN = svdNN.singularValues();
    double minSv = 1e10, maxSv = 0;
    for(int i=0;i<SNN.size();i++)
    {
        if(SNN[i] < minSv) minSv = SNN[i];
        if(SNN[i] > maxSv) maxSv = SNN[i];
    }
    for(int i=0;i<SNN.size();i++)
        { if(SNN[i] > setting_solverModeDelta*maxSv) SNN[i] = 1.0 / SNN[i]; else SNN[i] = 0; }

    MatXX Npi = svdNN.matrixU() * SNN.asDiagonal() * svdNN.matrixV().transpose(); 	// [dim] x 9.
    MatXX NNpiT = N*Npi.transpose(); 	// [dim] x [dim].
    MatXX NNpiTS = 0.5*(NNpiT + NNpiT.transpose());	// = N * (N' * N)^-1 * N'.

    if(b!=0) *b -= NNpiTS * *b;
    if(H!=0) *H -= NNpiTS * *H * NNpiTS;
}


// accumulates & shifts L.
void WindowedOptimizer::accumulateAF_MT(MatXX &H, VecX &b)
{

    if(mSystemSettings.SystemMultiThreading)// && false)
    {
        red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_A.get(), nFrames,  _1, _2, _3, _4), 0, 0, 0);
        red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<0>,
                accSSE_top_A.get(), &allEdges, this,  _1, _2, _3, _4), 0, allEdges.size(), 50);
        accSSE_top_A->stitchDoubleMT(red,H,b,this,false,true);
        resInA = accSSE_top_A->nres[0];
    }
    else
    {
        accSSE_top_A->setZero(nFrames);
        for(EFFrame* f : mEFrames)
            for(EFPoint* p : f->points)
                accSSE_top_A->addPoint<0>(p,this);
        I3D_LOG(i3d::detail) << "accSSE_top_A before stitch: " << accSSE_top_A->acc[0]->H;
        accSSE_top_A->stitchDoubleMT(red,H,b,this,false,false);
        I3D_LOG(i3d::detail) << "accSSE_top_A after stitch: " << accSSE_top_A->acc[0]->H;
        resInA = accSSE_top_A->nres[0];
    }
}

// accumulates & shifts L.
void WindowedOptimizer::accumulateLF_MT(MatXX &H, VecX &b)
{
    if(mSystemSettings.SystemMultiThreading)// && false)
    {
        red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_L.get(), nFrames,  _1, _2, _3, _4), 0, 0, 0);
        red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<1>,
                accSSE_top_L.get(), &allEdges, this,  _1, _2, _3, _4), 0, allEdges.size(), 50);
        accSSE_top_L->stitchDoubleMT(red,H,b,this,true,true);
        resInL = accSSE_top_L->nres[0];
    }
    else
    {
        accSSE_top_L->setZero(nFrames);
        for(EFFrame* f : mEFrames)
            for(EFPoint* p : f->points)
                accSSE_top_L->addPoint<1>(p,this);
        I3D_LOG(i3d::debug) << "accSSE_top_L before stitch: " << accSSE_top_L->acc[0]->H;
        accSSE_top_L->stitchDoubleMT(red,H,b,this,true,false);
        I3D_LOG(i3d::debug) << "accSSE_top_L after stitch: " << accSSE_top_L->acc[0]->H;

        resInL = accSSE_top_L->nres[0];
    }
}




//SC == SCHUR COMPLEMENT
void WindowedOptimizer::accumulateSCF_MT(MatXX &H, VecX &b)
{
    if(mSystemSettings.SystemMultiThreading)// && false)// && false)
    {
        red->reduce(boost::bind(&AccumulatedSCHessianSSE::setZero, accSSE_bot.get(), nFrames,  _1, _2, _3, _4), 0, 0, 0);
        red->reduce(boost::bind(&AccumulatedSCHessianSSE::addPointsInternal,
                accSSE_bot.get(), &allEdges, true,  _1, _2, _3, _4), 0, allEdges.size(), 0);
        accSSE_bot->stitchDoubleMT(red,H,b,this,true);
    }
    else
    {
        accSSE_bot->setZero(nFrames);
        for(EFFrame* f : mEFrames)
            for(EFPoint* p : f->points)
            {
                I3D_LOG(i3d::debug) << "Adding " << p->idxInPoints << " " << p->HdiF;
                accSSE_bot->addPoint(p, true);
            }
        accSSE_bot->stitchDoubleMT(red, H, b,this,false);
    }
}

void WindowedOptimizer::resubstituteFPt(const VecCf &xc, Mat16f* xAd, int min, int max, Vec10* stats, int tid)
{
    for(int k=min;k<max;k++)
    {
        EFPoint* p = allEdges.at(k);//[k];

        int nGoodRes = 0;
        //we can actually reduce the counting to the first "active" one
        for(EFResidual* r : p->residualsAll)
        {
            if(r->isActive()) 
            {    
                nGoodRes++;
                break;
            }
        }
        
        if(nGoodRes == 0)
        {
            p->data->step = 0;
            continue;
        }
        float b = p->bdSumF;
        b -= xc.dot(p->Hcd_accAF + p->Hcd_accLF);
        for(EFResidual* r : p->residualsAll)
        {
            if(!r->isActive()) continue;
            b -= xAd[r->hostIDX*nFrames + r->targetIDX] * r->JpJdF;
        }
        p->data->step = - b*p->HdiF;
        assert(std::isfinite(p->data->step));
    }
}
}
