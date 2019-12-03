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
#include "LocalMapper.h"
#include "Mapper.h"
#include "WindowedOptimizer.h"
#include "../Utils/timer.h"
#include "../Utils/CoarseDistanceMap.h"
#include "SystemSettings.h"
#include <opencv2/highgui.hpp>

namespace RESLAM
{

void 
LocalMapper::setOutputWrapper(const std::vector<IOWrap::Output3DWrapper*>& outputWrapper)
{
    std::copy(std::begin(outputWrapper),std::end(outputWrapper),std::back_inserter(mOutputWrapper));
}

LocalMapper::LocalMapper(const SystemSettings& systemSettings):
    mSystemSettings(systemSettings), mStopLocalMapping(false), Hcalib(systemSettings)
{ 
    mWindowedOptimizer = std::make_unique<WindowedOptimizer>(systemSettings);
    if (!mSystemSettings.LocalMapperLinearProcessing)
        localMappingThread =  std::thread(&LocalMapper::localMapping, this);
    if (mSystemSettings.LocalMapperUseCoarseDistanceMap)
    {   
        constexpr size_t EvalLvlCoarseDistMap{0}; 
        mCoarseDistanceMap = std::make_unique<CoarseDistanceMap>(Hcalib.width[EvalLvlCoarseDistMap],
                                                                 Hcalib.height[EvalLvlCoarseDistMap]);
    }
    I3D_LOG(i3d::info) << "setThreadReduce: " << &mThreadReduce;
    mWindowedOptimizer->setThreadReduce(&mThreadReduce);
}
LocalMapper::~LocalMapper()
{
    I3D_LOG(i3d::info) << "Closing the LocalMapper: ~LocalMapper";
    stopLocalMapping();
    I3D_LOG(i3d::info) << "Stopped the LocalMapper: ~LocalMapper";
    if (!mSystemSettings.LocalMapperLinearProcessing)
    {
         I3D_LOG(i3d::info) << "Waiting for join: ~LocalMapper";
        localMappingThread.join();
        I3D_LOG(i3d::info) << "Join complete: ~LocalMapper";
    }
    I3D_LOG(i3d::info) << "Closed the loop thread: ~LocalMapper";
}

void 
LocalMapper::queueKeyframe(FrameData* newKf)
{
    if (newKf == nullptr) return;
    {
        std::lock_guard<std::mutex> lock(mFrameHessiansQueueMutex);
        mFrameHessiansQueue.push(newKf);
        I3D_LOG(i3d::info) << "queue size: " << mFrameHessiansQueue.size();
    }
    mQueueCv.notify_one();
}

void
LocalMapper::addNewResidualsForOldPoints(FrameData* refFrameNew)
{
    // =========================== add new residuals for old points =========================
    I3D_LOG(i3d::info) << "addNewResidualsForOldPoints!";
    size_t numFwdResAdde = 0;
    for (auto& fh1: mFrameHessians)
    {
        if (refFrameNew == fh1) continue;
        I3D_LOG(i3d::info)<< "refFrameNew: " << refFrameNew << " fh1 " << fh1 << "fh->id: " << fh1->mKeyFrameId << " " 
                          << refFrameNew->mKeyFrameId << " " << refFrameNew->mFrameHeader->mFrameId << "goodE: " << fh1->getEdgesGood().size();
        for (auto& edgePixel : fh1->getEdgesGood())
        {
            I3D_LOG(i3d::debug) << "fh1: " << fh1->mKeyFrameId << " numFwdResAdde: " << numFwdResAdde
                               << "edgePixel->efPoint: " << edgePixel->efPoint;
            EdgeFrameResidual* edgeResidual = new EdgeFrameResidual(edgePixel, fh1, refFrameNew);
            edgeResidual->setState(ResState::IN);
            edgePixel->edgeResiduals.push_back(edgeResidual);
            mWindowedOptimizer->insertResidual(edgeResidual);
            edgePixel->lastResiduals[1] = edgePixel->lastResiduals[0];
            edgePixel->lastResiduals[0] = std::pair<EdgeFrameResidual*, ResState>(edgeResidual, ResState::IN);
            numFwdResAdde++;
        }
    }
}

double 
LocalMapper::calcLEnergy()
{
    if(setting_forceAcceptStep) return 0;

    const double Ef = mWindowedOptimizer->calcLEnergyF_MT();
    return Ef;

}

double 
LocalMapper::calcMEnergy()
{
    if(setting_forceAcceptStep) return 0;
    return mWindowedOptimizer->calcMEnergyF();

}

std::vector<VecX> 
LocalMapper::getNullspaces(std::vector<VecX> &nullspaces_pose, std::vector<VecX> &nullspaces_scale)
{
    nullspaces_pose.clear();
    nullspaces_scale.clear();

    int n=CPARS+mFrameHessians.size()*HessianSize;
    std::vector<VecX> nullspaces_x0_pre;
    for(int i=0;i<6;i++)
    {
        VecX nullspace_x0(n);
        nullspace_x0.setZero();
        for(const auto& fh : mFrameHessians)
        {
            nullspace_x0.segment<6>(CPARS+fh->idx*HessianSize) = fh->nullspaces_pose.col(i);
            nullspace_x0.segment<3>(CPARS+fh->idx*HessianSize) *= SCALE_XI_TRANS_INVERSE;
            nullspace_x0.segment<3>(CPARS+fh->idx*HessianSize+3) *= SCALE_XI_ROT_INVERSE;
        }
        nullspaces_x0_pre.push_back(nullspace_x0);
        nullspaces_pose.push_back(nullspace_x0);
    }

    VecX nullspace_x0(n);
    nullspace_x0.setZero();
    for(const auto& fh : mFrameHessians)
    {
        nullspace_x0.segment<6>(CPARS+fh->idx*HessianSize) = fh->nullspaces_scale;
        nullspace_x0.segment<3>(CPARS+fh->idx*HessianSize) *= SCALE_XI_TRANS_INVERSE;
        nullspace_x0.segment<3>(CPARS+fh->idx*HessianSize+3) *= SCALE_XI_ROT_INVERSE;
    }
    nullspaces_x0_pre.push_back(nullspace_x0);
    nullspaces_scale.push_back(nullspace_x0);

    return nullspaces_x0_pre;
}

void 
LocalMapper::solveSystem(int iteration, double lambda)
{
    mWindowedOptimizer->lastNullspaces_forLogging = getNullspaces(
            mWindowedOptimizer->lastNullspaces_pose,
            mWindowedOptimizer->lastNullspaces_scale);
    mWindowedOptimizer->solveSystemF(iteration, lambda, Hcalib);
}

void 
LocalMapper::backupState(bool backupLastStep)
{
    if(setting_solverMode & SOLVER_MOMENTUM)
    {
        if(backupLastStep)
        {
            Hcalib.step_backup = Hcalib.step;
            Hcalib.value_backup = Hcalib.value;
            for(auto* fh : mFrameHessians)
            {
                fh->step_backup = fh->step;
                fh->state_backup = fh->get_state();
                for(const auto& ph : fh->getEdgesGood())
                {
                    ph->backupIdepth();
                    ph->step_backup = ph->step;
                }
            }
        }
        else
        {
            Hcalib.step_backup.setZero();
            Hcalib.value_backup = Hcalib.value;
            for(auto* fh : mFrameHessians)
            {
                fh->step_backup.setZero();
                fh->state_backup = fh->get_state();
                for(const auto& ph : fh->getEdgesGood())
                {
                    ph->backupIdepth();
                    ph->step_backup = 0;
                }
            }
        }
    }
    else
    {
        Hcalib.value_backup = Hcalib.value;
        for(auto* fh : mFrameHessians)
        {
            fh->state_backup = fh->get_state();
            for(auto* ph : fh->getEdgesGood())
                ph->backupIdepth();
        }
    }
}

float
LocalMapper::optimize(size_t nOptIts)
{
    if(mFrameHessians.size() < 2) return 0;
    if(mFrameHessians.size() < 4) nOptIts = 15;
    if(mFrameHessians.size() < 3) nOptIts = 20;
	// get statistics and active residuals.
	mActiveResiduals.clear();
    int numPoints = 0;
    int numLRes = 0;
    int numRes = 0;
	for(FrameData* fh : mFrameHessians)
        //go through all the Edges
		for(EdgePixel* ph : fh->getEdgesGood())
		{
            //and then through all the residuals depending on this edge
			for(EdgeFrameResidual* r : ph->edgeResiduals)
			{
				if(!r->efResidual->isLinearized)
				{
					mActiveResiduals.push_back(r);
					r->resetOOB();
                    numRes++;
				}
				else
					numLRes++;
			}
			numPoints++;
		}

    if(!setting_debugout_runquiet)
        printf("OPTIMIZE %zu pts, %zu active res, %d lin res!\n",mWindowedOptimizer->getNbPoints(),mActiveResiduals.size(), numLRes);

    auto startTime = Timer::getTime();
    Vec3 lastEnergy = linearizeAll(false);
    double lastEnergyL = calcLEnergy();
    double lastEnergyM = calcMEnergy(); //Marginalized energy -> Maybe ignore for now
    auto endTime = Timer::getTime();
    I3D_LOG(i3d::error) << "Time for linearize and calcEnergy: " << Timer::getTimeDiffMiS(startTime,endTime) << " lastEnergy: " << lastEnergy.transpose();
    startTime = Timer::getTime();
	if(mSystemSettings.SystemMultiThreading)// && false)
		mThreadReduce.reduce(boost::bind(&LocalMapper::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, mActiveResiduals.size(), 50);
	else
		applyRes_Reductor(true,0,mActiveResiduals.size(),0,0);
    endTime = Timer::getTime();
    I3D_LOG(i3d::error) << "Time for applyRes_Reductor: " << Timer::getTimeDiffMiS(startTime,endTime);

    if(!setting_debugout_runquiet)
    {
        printf("Initial Error       \t");
    }
    double lambda{1e-1};
    float stepsize{1};
    VecX previousX = VecX::Constant(CPARS+ HessianSize*mFrameHessians.size(), NAN);
    LOG_THRESHOLD(i3d::info);
    I3D_LOG(i3d::info) << "Start Energy: " << lastEnergy.transpose() << " lastEnergyL: " << lastEnergyL;
    for(size_t iteration = 0; iteration < nOptIts; iteration++)
    {
        // solve!
        backupState(iteration!=0);
        solveSystem(iteration, lambda);
        I3D_LOG(i3d::info) << "Iteration: " << iteration << " lastX: " << mWindowedOptimizer->lastX.transpose();
        double incDirChange = (1e-20 + previousX.dot(mWindowedOptimizer->lastX)) / (1e-20 + previousX.norm() * mWindowedOptimizer->lastX.norm());
        previousX = mWindowedOptimizer->lastX;
        if(std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM))
        {
            float newStepsize = exp(incDirChange*1.4);
            if(incDirChange < 0 && stepsize > 1) stepsize=1;

            stepsize = std::sqrt(std::sqrt(newStepsize*stepsize*stepsize*stepsize));
            if(stepsize > 2) stepsize = 2;
            if(stepsize < 0.25) stepsize = 0.25;
        }

        bool canbreak = doStepFromBackup(stepsize,stepsize,stepsize,stepsize,stepsize);
        // eval new energy!
        const Vec3 newEnergy = linearizeAll(false);
        double newEnergyL = calcLEnergy();
        double newEnergyM = calcMEnergy();
        if(!setting_debugout_runquiet)
        {
            I3D_LOG(i3d::debug) << ((newEnergy[0] +  newEnergy[1] +  newEnergyL + newEnergyM <
                                   lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM) ? "ACCEPT" : "REJECT")
                                << " " << iteration << "(L: " << log10(lambda) << ", dir: " << incDirChange << "ss: " << stepsize;
        }
        I3D_LOG(i3d::info) << iteration << ": new Energy: " << newEnergy.transpose() << " newEnergyL " << newEnergyL << " lastEnergy: "
                            << lastEnergy.transpose() << " lastEnergyL: " << lastEnergyL
                            << "mActiveResiduals: " << mActiveResiduals.size();
        if(setting_forceAcceptStep || (newEnergy[0] +  newEnergy[1] +  newEnergyL + newEnergyM <
                lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM))
        {

            if(mSystemSettings.SystemMultiThreading)// && false)
                mThreadReduce.reduce(boost::bind(&LocalMapper::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, mActiveResiduals.size(), 50);
            else
                applyRes_Reductor(true,0,mActiveResiduals.size(),nullptr,0);
            lastEnergy = newEnergy;
            lastEnergyL = newEnergyL;
            lastEnergyM = newEnergyM;
            lambda *= 0.25;
        }
        else
        {
            loadStateBackup();
            lastEnergy = linearizeAll(false);
            lastEnergyL = calcLEnergy();
            lastEnergyM = calcMEnergy();
            lambda *= 1e2;
        }


        if(canbreak && iteration >= setting_minOptIterations) break;
    }
    I3D_LOG(i3d::info) << "Last Energy after break: " << lastEnergy.transpose() << " lastEnergyL: " << lastEnergyL;
    Vec6 newStateZero = Vec6::Zero();
    mFrameHessians.back()->setEvalPT(mFrameHessians.back()->getPRE_worldToCam(), newStateZero);
    EFDeltaValid=false;
    EFAdjointsValid=false;
    mWindowedOptimizer->setAdjointsF(Hcalib);
//    visualizeActiveResiduals();
    setPrecalcValues(); //ok
//    visualizeActiveResiduals();

    lastEnergy = linearizeAll(true);
    I3D_LOG(i3d::info) << "End Energy: " << lastEnergy.transpose() << " lastEnergyL: " << lastEnergyL;

    if(!std::isfinite(static_cast<double>(lastEnergy[0])) || !std::isfinite(static_cast<double>(lastEnergy[1])) || !std::isfinite(static_cast<double>(lastEnergy[2])))
    {
        printf("KF Tracking failed: LOST!\n");
        isLost=true;
    }



    statistics_lastFineTrackRMSE = std::sqrt(static_cast<float>(lastEnergy[0] / (mWindowedOptimizer->resInA)));
    I3D_LOG(i3d::info) << "statistics_lastFineTrackRMSE: " << statistics_lastFineTrackRMSE;
    LOG_THRESHOLD(i3d::info);
    return std::sqrt(static_cast<float>(lastEnergy[0] / (/*patternNum*ef*/mWindowedOptimizer->resInA)));
}

void 
LocalMapper::linearizeAll_Reductor(bool fixLinearization, std::vector<EdgeFrameResidual*>* toRemove, int min, int max, Vec10* stats, int tid)
{
    std::vector<size_t> nRes(mFrameHessians.size());
    std::fill(nRes.begin(),nRes.end(),0);
    I3D_LOG(i3d::detail) << "linearizeAll_Reductor: min/max: " << min << "/" << max << " = " << (max-min) << ", tid:" << tid;
    auto nRemoved = 0;
    for(int k=min;k<max;k++)
    {
        EdgeFrameResidual* r = mActiveResiduals[k];
        (*stats)[0] += r->linearize(Hcalib,mSystemSettings);
        //debug
        if (r->state_NewState == ResState::IN)
        {
            nRes[r->target->idx]++;
        }
        else nRemoved++;
        if(fixLinearization)
        {
            r->applyRes(true);
            if(r->efResidual->isActive())
            {
                if(r->isNew)
                {
                    EdgePixel* p = r->edgePixel;
                    const auto& targetPrecalc = r->host->getTargetPrecalcById(r->target->idx);
                    const Vec3f ptp_inf = targetPrecalc.PRE_KRKiTll * Vec3f(p->hostX,p->hostY, 1);	// projected point assuming infinite depth.
                    const Vec3f ptp = ptp_inf + targetPrecalc.PRE_KtTll*p->getIdepthScaled();	// projected point with real depth.
                    float relBS = 0.01*((ptp_inf.head<2>() / ptp_inf[2])-(ptp.head<2>() / ptp[2])).norm();	// 0.01 = one pixel.
                    if(relBS > p->maxRelBaseline)
                        p->maxRelBaseline = relBS;
                    p->numGoodResiduals++;
                }
            }
            else
            {
                toRemove[tid].push_back(mActiveResiduals[k]);
            }
        }
    }
}

bool 
LocalMapper::doStepFromBackup(float stepfacC,float stepfacT,float stepfacR,float stepfacA,float stepfacD)
{
    Vec6 pstepfac;
    pstepfac.segment<3>(0).setConstant(stepfacT);
    pstepfac.segment<3>(3).setConstant(stepfacR);
    float sumT=0, sumR=0, numID=0; 
    float sumNID=0;
    if(setting_solverMode & SOLVER_MOMENTUM)
    {
        Hcalib.setValue(Hcalib.value_backup + Hcalib.step);
        for(auto* fh : mFrameHessians)
        {
            Vec6 step = fh->step;
            step.head<6>() += 0.5f*(fh->step_backup.head<6>());

            fh->setState(fh->state_backup + step);
            I3D_LOG(i3d::info) << "kfId: " << fh->mKeyFrameId << "Making step: " << fh->state_backup << " + " << step << " = " << fh->state_backup+step;

            sumT += step.segment<3>(0).squaredNorm();
            sumR += step.segment<3>(3).squaredNorm();

            for(const auto& ph : fh->getEdgesGood())
            {
                float step = ph->step+0.5f*(ph->step_backup);
                if (!mSystemSettings.LocalMapperOptimizeDepth) step = 0;
                const auto idepth_backup = ph->getIdepthBackup();
                ph->setIdepth(idepth_backup + step);
                sumNID += std::abs(idepth_backup);
                numID++;
                ph->setIdepthZero(idepth_backup + step);
            }
        }
    }
    else
    {
        Hcalib.setValue(Hcalib.value_backup + stepfacC*Hcalib.step);
        for(auto* fh : mFrameHessians)
        {
            fh->setState(fh->state_backup + pstepfac.cwiseProduct(fh->step));
            sumT += fh->step.segment<3>(0).squaredNorm();
            sumR += fh->step.segment<3>(3).squaredNorm();
            I3D_LOG(i3d::detail) << fh->mKeyFrameId << "Making step: " << fh->state_backup.transpose() << " + " << pstepfac.cwiseProduct(fh->step).transpose() << " = " << (fh->state_backup+pstepfac.cwiseProduct(fh->step)).transpose();
            for(EdgePixel* ph : fh->getEdgesGood())
            {

                if (!mSystemSettings.LocalMapperOptimizeDepth) ph->step = 0;
                const auto idepth_backup = ph->getIdepthBackup();
                ph->setIdepth(idepth_backup + stepfacD*ph->step);
                sumNID += fabsf(idepth_backup);
                numID++;
                ph->setIdepthZero(idepth_backup + stepfacD*ph->step);
            }
        }
    }

    sumR /= mFrameHessians.size();
    sumT /= mFrameHessians.size();
    // sumID /= numID;
    sumNID /= numID;



        I3D_LOG(i3d::debug) <<"; R " <<
                std::sqrt(sumR) / (0.00005*setting_thOptIterations) <<"; T" <<
                std::sqrt(sumT)*sumNID / (0.00005*setting_thOptIterations);

    EFDeltaValid=false;
    setPrecalcValues(); //OK

    return  std::sqrt(sumR) < 0.00005*setting_thOptIterations &&
            std::sqrt(sumT)*sumNID < 0.00005*setting_thOptIterations;
}

Vec3 
LocalMapper::linearizeAll(bool fixLinearization)
{
    double lastEnergyP = 0;
    double lastEnergyR = 0;
    double num = 0;
    I3D_LOG(i3d::info) << " linearizeAll! " << mActiveResiduals.size();

    std::vector<EdgeFrameResidual*> toRemove[NumThreads];
    for(size_t i = 0; i < NumThreads; ++i) 
        toRemove[i].clear();

    if(mSystemSettings.SystemMultiThreading)
    {
        mThreadReduce.stats.setZero();
        mThreadReduce.reduce(boost::bind(&LocalMapper::linearizeAll_Reductor, this, fixLinearization, toRemove, _1, _2, _3, _4), 0, mActiveResiduals.size(), 0);
        lastEnergyP = mThreadReduce.stats[0];
    }
    else
    {
        if (mActiveResiduals.size()>0)
        {
           auto r = mActiveResiduals[0];
           I3D_LOG(i3d::debug) << "Calling " << r->host->idx << " to " << r->target->idx
                              << " with PRE_KRKiTll: " << r->host->getTargetPrecalcById(r->target->idx).PRE_KRKiTll << " and "
                              << r->host->getTargetPrecalcById(r->target->idx).PRE_tTll.transpose();
        }

        Vec10 stats = Vec10::Zero();// stats.setZero();
        linearizeAll_Reductor(fixLinearization, toRemove, 0, mActiveResiduals.size(),&stats,0);
        lastEnergyP = stats[0];
    }

    if(fixLinearization)
    {

        for(auto* r : mActiveResiduals)
        {
            auto* ph = r->edgePixel;
            if(ph->lastResiduals[0].first == r)
                ph->lastResiduals[0].second = r->state_state;
            else if(ph->lastResiduals[1].first == r)
                ph->lastResiduals[1].second = r->state_state;
        }

        for(size_t i = 0; i < NumThreads; i++)
        {
            for(auto* r : toRemove[i])
            {
                auto* ph = r->edgePixel;

                if(ph->lastResiduals[0].first == r)
                    ph->lastResiduals[0].first = nullptr;
                else if(ph->lastResiduals[1].first == r)
                    ph->lastResiduals[1].first = nullptr;

                const auto resIt = std::find(std::begin(ph->edgeResiduals),std::end(ph->edgeResiduals), r);
                                    // [&r](EdgeFrameResidual* res){ return res == r;});
                I3D_LOG(i3d::detail) << "std::find at k == " << std::distance(std::begin(ph->edgeResiduals),resIt);
                if (resIt != std::end(ph->edgeResiduals))
                {
                    I3D_LOG(i3d::detail) << "deleting at: " << r->efResidual << " and " << *resIt;
                    mWindowedOptimizer->dropResidual(r->efResidual);
                    ph->edgeResiduals.erase(resIt);
                }
            }
        }
    }

    return Vec3(lastEnergyP, lastEnergyR, num);
}

void
LocalMapper::applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid)
{
    for(int k = min; k < max; k++)
    {
        //TODO: rewrite to without boundary check -> [k]
        mActiveResiduals.at(k)->applyRes(true);
    }
}

void
LocalMapper::printWindowPoses() const
{
    //TODO: This method is for debugging! Delete that method!!
    return;
    for (const auto& kf : mFrameHessians)
    {
        I3D_LOG(i3d::info) << "idx: " << kf->mKeyFrameId << "/" << kf->mFrameHeader->mFrameId
                           << " poseUpd: " << kf->getPRE_camToWorld().matrix3x4() << "/"
                           << " pose: " << kf->mFrameHeader->getCamToWorld().matrix3x4();
    }
}

void 
LocalMapper::localMappingLinear(FrameData* newKf)
{
    if (newKf == nullptr) return;
    std::lock_guard<std::mutex> lock(mFrameHessiansMutex);
    printWindowPoses();
    // =========================== Flag Frames to be Marginalized. =========================
    flagFramesForMarginalization(newKf);
    newKf->setEvalPT_scaled(newKf->mFrameHeader->getWorldToCam());//   camToWorld.inverse()); 

    // =========================== add New Frame to Hessian Struct. =========================
    addKeyFrame(newKf);
    constexpr size_t nFixedPoses{0};
    for (auto& f : mFrameHessians)
        f->FIX_WORLD_POSE = false;

    for (size_t idx = 0; idx < mFrameHessians.size() && idx < nFixedPoses; ++idx)
    {
        I3D_LOG(i3d::info) << "Fixing: " << mFrameHessians[idx]->mKeyFrameId;
        mFrameHessians[idx]->FIX_WORLD_POSE = true;
    }

    setPrecalcValues();
    addNewResidualsForOldPoints(newKf);
    // =========================== Activate Points (& flag for marginalization). =========================
    activatePointsMT();
    mWindowedOptimizer->makeIDX();

	// =========================== OPTIMIZE ALL =========================
    I3D_LOG(i3d::info) << "Window poses before optimize:";
    printWindowPoses();
	const float rmse = optimize(setting_maxOptIterations);
    I3D_LOG(i3d::info) << "Window poses after optimize:";
    printWindowPoses();
    I3D_LOG(i3d::info) << "RMSE: " << rmse;
    
    // =========================== REMOVE OUTLIER =========================
    removeOutliers();

    // =========================== (Activate-)Marginalize Points =========================
    flagPointsForRemoval();
    I3D_LOG(i3d::info) << "dropPointsF after flagPointsForRemoval!";
    mWindowedOptimizer->dropPointsF();
    getNullspaces(mWindowedOptimizer->lastNullspaces_pose, mWindowedOptimizer->lastNullspaces_scale);
    //At the moment we do not marginalize!
    mWindowedOptimizer->marginalizePointsF();

    for(auto* ow : mOutputWrapper)
    {
        ow->publishGraph(mWindowedOptimizer->mConnectivityMap);
        ow->publishKeyframes(mFrameHessians, Hcalib);
    }

    // =========================== Marginalize Frames =========================
    Hcalib.makeK();
    I3D_LOG(i3d::info) << "Window poses before updatePosesAfterLocalMapper:";
    printWindowPoses();
    if (mSystemSettings.LocalMapperUseNewPoseUpdate)
        mParentMapper->updatePosesAfterLocalMapperNew(mFrameHessians);
    else
        mParentMapper->updatePosesAfterLocalMapper(mFrameHessians);

    I3D_LOG(i3d::info) << "Window poses after updatePosesAfterLocalMapper:";
    printWindowPoses();
    for(size_t i=0; i < mFrameHessians.size(); i++)
    {   
        if(mFrameHessians[i]->mFlaggedForMarginalization)
        {
            
            marginalizeFrame(mFrameHessians[i]);
            i = 0; //I'm not really happy with resetting i
        }
    }
    publishGraph();
}

void
LocalMapper::flagPointsForRemoval()
{
    assert(EFIndicesValid);
    std::vector<FrameData*> fhsToKeepPoints;
    std::vector<FrameData*> fhsToMargPoints;

    std::copy_if(mFrameHessians.begin(),mFrameHessians.end(),std::back_inserter(fhsToMargPoints),[](const auto& fh){return fh->mFlaggedForMarginalization; });
    std::copy_if(mFrameHessians.begin(),mFrameHessians.end(),std::back_inserter(fhsToKeepPoints),[](const auto& fh){return !fh->mFlaggedForMarginalization; });

    I3D_LOG(i3d::info) << "before: removeOutOfBounds";
    // LOG_THRESHOLD(i3d::debug);
    for (auto* host : mFrameHessians)
    {
        host->removeOutOfBounds(*mWindowedOptimizer,Hcalib,fhsToMargPoints,mSystemSettings);
    }
    // LOG_THRESHOLD(i3d::info);
    I3D_LOG(i3d::info) << "after: removeOutOfBounds";
}

/**
 * Removes outlier edges, i.e. edges without residuals
 */
void LocalMapper::removeOutliers()
{
    for(FrameData* fh : mFrameHessians)
    {
        I3D_LOG(i3d::detail) << "removeOutliers fh " << fh->idx << ", " << fh->mKeyFrameId << " before delete: " << fh->getEdgesGood().size() << " edgesOut: " << fh->getEdgesOut().size();
        const auto numPointsDropped = fh->removeGoodEdgesWithoutResiduals();
        I3D_LOG(i3d::detail) << "removeOutliers fh " << fh->idx << ", " << fh->mKeyFrameId << " after delete: " << fh->getEdgesGood().size() 
                             << " edgesOut: " << fh->getEdgesOut().size() << "numPointsDropped: " << numPointsDropped;
    }
    I3D_LOG(i3d::info) << "dropPointsF in removeOutliers!";
    mWindowedOptimizer->dropPointsF();
}

/**
 * When performing loop closure, we fix the poses of the current window and optimize all the other poses.
 * Note that this typically results in the origin not being at identity.
 */
void LocalMapper::fixWindowPoses(CeresPoseVector& ceresPoseVector) const
{
    std::lock_guard<std::mutex> l(mFrameHessiansMutex);
    for (const auto& kf : mFrameHessians)
    {
        ceresPoseVector[kf->mKeyFrameId].setPoseFixed();
    }
}

/**
 * Adds constraints between keyframes in the window
 */
void LocalMapper::addWindowConstraints(RelPoseConstraints& ceresConstraints)
{
    std::lock_guard<std::mutex> l(mFrameHessiansMutex);
    I3D_LOG(i3d::info) << "Adding constraints from windowed BA" << mFrameHessians.size();
    size_t nConstraints = 0;
    for (size_t idx = 0; idx < mFrameHessians.size();++idx)
    {
        const auto& f1 = mFrameHessians[idx];
        const SE3Pose T_W_idx = f1->mFrameHeader->getCamToWorld();
        const auto id1 = f1->mKeyFrameId; 
        for (size_t idx2 = idx+1; idx2 < mFrameHessians.size() && (idx2-idx) <= mSystemSettings.LoopClosureNumberOfBAConstraints;++idx2)
        {
            const auto& f2 = mFrameHessians[idx2];
            const SE3Pose T_id2_id1 = f2->mFrameHeader->getWorldToCam()*T_W_idx;
            I3D_LOG(i3d::info) << "Emplacing window constraint: " << id1 << " to " << f2->mKeyFrameId;
            ceresConstraints.emplace_back(T_id2_id1,id1,f2->mKeyFrameId);
            nConstraints++;
        }
    }
    I3D_LOG(i3d::info) << "Added " << nConstraints << " constraints from windowed BA";
}

/**
 * Marginalize a frame and delete all corresponding observations
 */
void LocalMapper::marginalizeFrame(FrameData* frame)
{
    // marginalize or remove all this frame's points.
    I3D_LOG(i3d::info) << "marginalizeFrame: " << frame->mKeyFrameId << " data: " << frame->getEdgesGood().size();;
    assert((int)frame->getEdgesGood().size()==0);
    mWindowedOptimizer->marginalizeFrame(frame->efFrame);

    // drop all observations of existing edges in the frame to be marginalized
    for(FrameData* fh : mFrameHessians)
    {
        if (fh == frame) continue;
        for(EdgePixel* ph : fh->getEdgesGood())
        {
            auto resIt = std::find_if(ph->edgeResiduals.begin(),ph->edgeResiduals.end(),
                        [&](const EdgeFrameResidual* r){ return r->target == frame; });                
            if (resIt != ph->edgeResiduals.end())
            {
                I3D_LOG(i3d::detail) << "New found at: " << std::distance(ph->edgeResiduals.begin(),resIt) << " with: " << (*resIt)->target->mKeyFrameId << "*resIt = " << *resIt;
                auto* r = *resIt;               
                if(ph->lastResiduals[0].first == r)
                    ph->lastResiduals[0].first = nullptr;
                else
                {
                    if(ph->lastResiduals[1].first == r)
                        ph->lastResiduals[1].first = nullptr;
                }
                if(r->host->mKeyFrameId < r->target->mKeyFrameId)
                    statistics_numForceDroppedResFwd++;
                else
                    statistics_numForceDroppedResBwd++;
                mWindowedOptimizer->dropResidual(r->efResidual);
                ph->edgeResiduals.erase(resIt);
            }
        }
    }
    frame->resetFrame();
    frame->mFrameHeader->marginalizedAt = mFrameHessians.back()->mFrameHeader->mFrameId;
    frame->mFrameHeader->movedByOpt = frame->w2c_leftEps().norm();
    if (mSystemSettings.EnableLoopClosure)
    {
        //Add the constraints before deleting the frame
        I3D_LOG(i3d::info) << "Marginalizing frame: " << frame->mKeyFrameId << " with idx: " << frame->idx 
                           << " and allID: " << frame->mFrameHeader->mFrameId
                           << " frame is in DB: " << frame->isInFernDb() << " inloop: " << frame->isInLoop();
        const SE3Pose T_W_marg = frame->getPRE_camToWorld(); //transformation from frame to marginalize to the world
        size_t nConstraints{0};
        for (size_t idx = 0; idx < mFrameHessians.size() && nConstraints < mSystemSettings.LoopClosureNumberOfBAConstraints; ++idx)
        {
            const auto& fh = mFrameHessians[idx];    
            I3D_LOG(i3d::info) << fh->mKeyFrameId << "<->" << frame->mKeyFrameId << " isInFernDb: " << fh->isInFernDb() << ", inLoop: " << fh->isInLoop(); 
            if (fh != frame)
            {
                const size_t diff = fh->idx < frame->idx ? frame->idx - fh->idx : fh->idx - frame->idx; //basically std::abs for size_t
                if ((fh->isInFernDbOrLoop() && mSystemSettings.LoopClosureOnlyUseFernKF && frame->isInFernDbOrLoop()) || 
                    (diff <= mSystemSettings.LoopClosureNumberOfBAConstraints && !mSystemSettings.LoopClosureOnlyUseFernKF))
                {
                    frame->addConstraint(CeresConstraint(fh->getPRE_worldToCam()*T_W_marg,frame->mKeyFrameId,fh->mKeyFrameId));
                    I3D_LOG(i3d::info) << "Adding constraint " << nConstraints << " for " << frame->mKeyFrameId << "->"<<fh->mKeyFrameId;
                    ++nConstraints;
                    mMargConnectivityGraph[(((uint64_t)frame->mKeyFrameId) << 32) + ((uint64_t)fh->mKeyFrameId)] = Vec2i(1,1);
                    mMargConnectivityGraph[(((uint64_t)fh->mKeyFrameId) << 32) + ((uint64_t)frame->mKeyFrameId)] = Vec2i(1,1);
                }
            }
        }
        //We have a problem, since no frame in the window is in a loop or in the db
        if (nConstraints == 0 && (mSystemSettings.LoopClosureOnlyUseFernKF && frame->isInFernDbOrLoop()))
        {
            I3D_LOG(i3d::info) << frame->mKeyFrameId << " not enough constraints!";
            const FrameData* const newRefFrame = mParentMapper->findFirstValidKeyframe(frame);
            frame->addConstraint(CeresConstraint(newRefFrame->getPRE_worldToCam()*T_W_marg,frame->mKeyFrameId,newRefFrame->mKeyFrameId));
            mMargConnectivityGraph[(((uint64_t)frame->mKeyFrameId) << 32) + ((uint64_t)newRefFrame->mKeyFrameId)] = Vec2i(1,1);
            mMargConnectivityGraph[(((uint64_t)newRefFrame->mKeyFrameId) << 32) + ((uint64_t)frame->mKeyFrameId)] = Vec2i(1,1);
            I3D_LOG(i3d::info) << frame->mKeyFrameId << " new constraints " << newRefFrame->mKeyFrameId
                               << "pose: " << newRefFrame->getPRE_worldToCam().matrix3x4()
                               << "pose: " << newRefFrame->mFrameHeader->getWorldToCam().matrix3x4();
        }
        else
        {
            I3D_LOG(i3d::info) << frame->mKeyFrameId << " with: " << nConstraints << " enough constraints!";
        }
    }

    if (mSystemSettings.LoopClosureOnlyUseFernKF && mSystemSettings.EnableLoopClosure)
    {
        if (!frame->isInFernDbOrLoop())
        {
            I3D_LOG(i3d::info) << "Searching for new real kf: " << frame->mFrameHeader->mFrameId << ": " << frame->isInFernDb() << " and " << frame->isInLoop();
            //We find the closest frame before this one that is a real kf and make it the new reference frame
            const FrameData* const newRefFrame = mParentMapper->findLastValidKeyframe(frame);
            I3D_LOG(i3d::info) << "Keyframe found: " << newRefFrame->mKeyFrameId << " allId: " << newRefFrame->mFrameHeader->mFrameId;
            const SE3Pose camToRef = newRefFrame->getPRE_worldToCam()*frame->getPRE_camToWorld(); //ok
            I3D_LOG(i3d::info) << "camToRef Before: " << frame->mFrameHeader->getCamToRef().matrix3x4()
                               << " camToRef after: " << camToRef.matrix3x4()
                               << " worldPose: " << frame->getPRE_camToWorld().matrix3x4()
                               << " after: " << (newRefFrame->getPRE_camToWorld()*camToRef).matrix3x4()
                               << " worldPose of new: " << newRefFrame->getPRE_camToWorld().matrix3x4();
            //Make it to a "normal" frame
            frame->mFrameHeader->setRefFrame(newRefFrame);
            frame->mFrameHeader->setCamToRef(camToRef);
            frame->mFrameHeader->updateCamToWorld(newRefFrame->getPRE_camToWorld()*camToRef); //same as setting it to it's world pose!
            I3D_LOG(i3d::info) << "camToRef after: " << frame->mFrameHeader->getCamToRef().matrix3x4(); 
            frame->mFrameHeader->mIsKeyFrame = false;
            
            mParentMapper->changeRefFrameLock(frame,newRefFrame);
            I3D_LOG(i3d::info) << "camToRef " << frame->mFrameHeader->mFrameId << " after changeRefFrame: " << frame->mFrameHeader->getCamToRef().matrix3x4(); 

            I3D_LOG(i3d::info) << "Making kf : " << frame->mKeyFrameId << " to normal frame with refFrame: " << newRefFrame->mKeyFrameId
                               << " frame id kf: " << frame->mFrameHeader->mFrameId << " newref id: " << newRefFrame->mFrameHeader->mFrameId;
        }
    }
    I3D_LOG(i3d::info) << "Marginalizing frame: " << frame->mKeyFrameId << " size: " << mFrameHessians.size();
    mFrameHessians.erase(std::find(mFrameHessians.begin(),mFrameHessians.end(),frame));
    I3D_LOG(i3d::info) << " After marginalizing frame size: " << mFrameHessians.size();
    //do not delete the keyframe!
    for(size_t i = 0; i < mFrameHessians.size();i++)
        mFrameHessians[i]->idx = i;
    setPrecalcValues();
    mWindowedOptimizer->setAdjointsF(Hcalib);
}

void 
LocalMapper::setPrecalcValues()
{
    for (FrameData* fh : mFrameHessians)
        fh->computeTargetPrecalc(mFrameHessians, Hcalib);
    mWindowedOptimizer->setDeltaF(Hcalib);
}


/**
 * Adds a keyframe to the local map and the windowed optimizer
 */
void LocalMapper::addKeyFrame(FrameData* refFrameNew)
{
    refFrameNew->idx = mFrameHessians.size();
    mFrameHessians.push_back(refFrameNew);
    mWindowedOptimizer->insertKeyFrame(refFrameNew,Hcalib);
}

/**
 * Waits in a separate thread for new keyframes to arrive 
 */
void LocalMapper::localMapping()
{
    while (!hasStoppedLocalMapping())
    {
        I3D_LOG(i3d::info) << "localMapping!";
        FrameData* newestKf = nullptr;
        {
            std::unique_lock<std::mutex> lock(mFrameHessiansQueueMutex);
            //Wait for signal or process if we have an element in queue
            mQueueCv.wait(lock,[this](){return !mFrameHessiansQueue.empty() || hasStoppedLocalMapping();});
            if (hasStoppedLocalMapping()) 
            {
                I3D_LOG(i3d::info) << "Break loop in local Mapping!";
                break;
            }
            newestKf = std::move(mFrameHessiansQueue.front());
            mFrameHessiansQueue.pop();
        }
        if (newestKf != nullptr)
        {
            //We got a new frame
            localMappingLinear(newestKf);
        }
    }
}

void
LocalMapper::computeCurrentMinActDist()
{
    if (!mSystemSettings.LocalMapperAdaptMinActDist) return;
    // constexpr float setting
    if(mWindowedOptimizer->getNbPoints() < setting_desiredPointDensity*0.66)
		mCurrentMinActDst -= 0.8;
	if(mWindowedOptimizer->getNbPoints() < setting_desiredPointDensity*0.8)
		mCurrentMinActDst -= 0.5;
	else if(mWindowedOptimizer->getNbPoints() < setting_desiredPointDensity*0.9)
		mCurrentMinActDst -= 0.2;
	else if(mWindowedOptimizer->getNbPoints() < setting_desiredPointDensity)
		mCurrentMinActDst -= 0.1;

	if(mWindowedOptimizer->getNbPoints() > setting_desiredPointDensity*1.5)
		mCurrentMinActDst += 0.8;
	if(mWindowedOptimizer->getNbPoints() > setting_desiredPointDensity*1.3)
		mCurrentMinActDst += 0.5;
	if(mWindowedOptimizer->getNbPoints() > setting_desiredPointDensity*1.15)
		mCurrentMinActDst += 0.2;
	if(mWindowedOptimizer->getNbPoints() > setting_desiredPointDensity)
		mCurrentMinActDst += 0.1;

	if(mCurrentMinActDst < 0) mCurrentMinActDst = 0;
	if(mCurrentMinActDst > 4) mCurrentMinActDst = 4;
}

void 
LocalMapper::activateEdges_Reductor(EdgePixelVec* optimized, DetectedEdgesPtrVec* toOptimize,int min, int max, Vec10* stats, int tid)
{
    ImmatureEdgeTemporaryResidual* tr = new ImmatureEdgeTemporaryResidual[mFrameHessians.size()];
    I3D_LOG(i3d::detail) << "activateEdges_Reductor min/max: " << min << " " << max;
    size_t nres = 0;
    for(int k = min; k < max; ++k)
    {
        (*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k],1,tr);
        if ((*optimized)[k] != nullptr) ++nres;
    }
    I3D_LOG(i3d::info) << "Added: " << nres << " out of " << (*toOptimize).size() << " immature points to active points";
    delete[] tr;
}

void 
LocalMapper::activatePointsMT()
{
    FrameData* refFrameNew = mFrameHessians.back();
    if (mSystemSettings.LocalMapperUseCoarseDistanceMap)
    {
        computeCurrentMinActDist();
        if (mCoarseDistanceMap != nullptr)
        {
            mCoarseDistanceMap->makeK(Hcalib);
            I3D_LOG(i3d::info) << "Making new distance map for refFrameNew" << refFrameNew->mKeyFrameId;
            mCoarseDistanceMap->makeDistanceMap(mFrameHessians, refFrameNew);
        }
    }
    //Now, we have to check, if we can activate new points!
    //We always try to activate as many as possible!
    //This essentially generates EdgePixels from UncertainPoints
    DetectedEdgesPtrVec newEdgePixels; //same as toOptimize in DSO
    newEdgePixels.reserve(50000);
    for (auto* fh1: mFrameHessians)
    {
        if (refFrameNew == fh1) continue;
        assert(refFrameNew->mFrameHeader->mFrameId != fh1->mFrameHeader->mFrameId && "Ids shouldn't be equal!");
        // if (!fh1->mFlaggedForMarginalization)
        const float compareDist = (mSystemSettings.LocalMapperAdaptMinActDist ? mCurrentMinActDst : mSystemSettings.LocalMapperMinDistInDistMapForValidPixels);
        fh1->findResidualsToOptimize(newEdgePixels,mCoarseDistanceMap.get(),refFrameNew ,Hcalib, mSystemSettings, compareDist);
    }
    I3D_LOG(i3d::info) << "allKeyFrameData.size(): " << mFrameHessians.size() << " " << newEdgePixels.size();

    //activate points!
    std::vector<EdgePixel* > optEdgePixels; //optimized
    optEdgePixels.resize(newEdgePixels.size());
    //checked till here
    I3D_LOG(i3d::debug) << "optEdgePixels.size: " << optEdgePixels.size() << " newEdgePixels: " << newEdgePixels.size();
    if (mSystemSettings.SystemMultiThreading)// && false)
        mThreadReduce.reduce(boost::bind(&LocalMapper::activateEdges_Reductor, this, &optEdgePixels, &newEdgePixels, _1, _2, _3, _4), 0, optEdgePixels.size(), 50);
    else
    {
        activateEdges_Reductor(&optEdgePixels,&newEdgePixels,0,newEdgePixels.size(),nullptr,0);
    }
    I3D_LOG(i3d::info) << "optEdgePixels.size: " << optEdgePixels.size() << " newEdgePixels: " << newEdgePixels.size();// << " statistics_numActivatedPoints: " << statistics_numActivatedPoints;
    size_t nGood = 0;
    for (size_t i = 0; i < newEdgePixels.size(); ++i)
    {
        auto* newEdge = optEdgePixels[i];
        I3D_LOG(i3d::detail) << "i: " << i << " newEdge: " << newEdge;
        if (newEdge != nullptr)
        {
            auto* validEdge = newEdgePixels[i];
            validEdge->flagAlreadyInWindow = true;
            //Instead of removing the detected edges, we "mark" them as good edges
            newEdge->host->addGoodEdge(newEdge);
            mWindowedOptimizer->insertEdgePixel(newEdge);
            I3D_LOG(i3d::debug) << "adding newEdge with " << newEdge << " and " << newEdge->edgeResiduals.size();
            for (auto* r : newEdge->edgeResiduals)
                mWindowedOptimizer->insertResidual(r);
            nGood++;
        }
    }
    size_t nUncertAft{0};
    for (auto& fh1: mFrameHessians) 
    {
        fh1->cleanUpPointsAlreadyInWindow();
        nUncertAft += fh1->getDetectedEdges().size();
    }

}

void 
LocalMapper::flagFramesForMarginalization(FrameData* newFH)
{
    LOG_THRESHOLD(i3d::info);
    I3D_LOG(i3d::info) << "flagFramesForMarginalization" << mFrameHessians.size();


    size_t nFramesFlagged{0};
    // marginalize all frames that have not enough points.
    for (auto& fh : mFrameHessians)
    {
        const auto in = fh->getEdgesGood().size() + fh->getDetectedEdges().size();
        const auto out = fh->getEdgesMarginalized().size() + fh->getEdgesOut().size();

        if((in < LMS::MinPtsRemainingForMarg * (in+out)) && (mFrameHessians.size() > LMS::MinFrames+nFramesFlagged))
        {
            fh->mFlaggedForMarginalization = true;
            nFramesFlagged++;
        }
    }

    // marginalize one.
    //If we have too many frames
    if(mFrameHessians.size() >= LMS::MaxFrames + nFramesFlagged)
    {
        double smallestScore = 1;
        FrameData* toMarginalize = nullptr;
        const FrameData* latest = mFrameHessians.back();

        
        for(auto* fh : mFrameHessians)
        {
            if((fh->mKeyFrameId + LMS::MinFrameAge) > (latest->mKeyFrameId ) || fh->mKeyFrameId == 0) continue;

            const auto distScore = fh->computeTargetPrecalcDistScore(latest);
            I3D_LOG(i3d::fatal) << "frameIdAll: " << fh->mFrameHeader->mFrameId << " keyframeId: " << fh->mKeyFrameId << " score: " << distScore;
            if(distScore < smallestScore)
            {
                smallestScore = distScore;
                toMarginalize = fh;
            }
        }

        toMarginalize->mFlaggedForMarginalization = true;
        ++nFramesFlagged;
    }
}

EdgePixel* 
LocalMapper::optimizeImmaturePoint(const DetectedEdge* point, int minObs, ImmatureEdgeTemporaryResidual* residuals)
{
    LOG_THRESHOLD(i3d::nothing);
    if (point->flagAlreadyInWindow)
    {
        I3D_LOG(i3d::info) << "Not adding: " << point->flagAlreadyInWindow;
        exit(0);
        return nullptr;
    } 
    size_t nRes{0};
    for (FrameData* fh : mFrameHessians)
    {
        if (fh != point->host) //if target != host
        {
            residuals[nRes].state_NewEnergy = residuals[nRes].state_energy = 0;
            residuals[nRes].state_NewState = ResState::OUTLIER;
            residuals[nRes].state_state = ResState::IN;
            residuals[nRes].target = fh;
            nRes++;
        }
    }
    I3D_LOG(i3d::debug) << "nRes: " << nRes << " point: " << point->getPixel2D().transpose();
    bool print = false;//rand()%50==0;
    float lastEnergy{0}, lastHdd{0}, lastbd{0};
    float currentIdepth = point->idepth;
    for(size_t i = 0; i < nRes; ++i)
    {
        lastEnergy += point->linearizeResidual(Hcalib, 1000, residuals+i,lastHdd, lastbd, currentIdepth,mSystemSettings);
        residuals[i].state_state = residuals[i].state_NewState;
        residuals[i].state_energy = residuals[i].state_NewEnergy;
    }
    //Activate as many points as possible!
    auto activateButDontOptimize = false;
    //check till here
    if(!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act) //setting_minIdepthH_act?
    {
        if (std::isfinite(lastEnergy))
        {
            //Point was initialized but depth is bad!
            activateButDontOptimize = true;
        }
        else
        {
            I3D_LOG(i3d::info) <<"lastHdd < 100 OptPoint: Not well-constrained"<<point->getPixel2DHom().transpose();

            return nullptr;
        }
    }
    if(print) printf("Activate point. %zd residuals. H=%f. Initial Energy: %f. Initial Id=%f\n" ,
            nRes, lastHdd,lastEnergy,currentIdepth);


    float lambda = 0.1;
    for(auto iteration = 0; iteration < 3 && !activateButDontOptimize; iteration++)
    {
        float H = lastHdd;
        H *= 1+lambda;
        float step = (1.0/H) * lastbd;
        float newIdepth = currentIdepth - step;

        float newHdd = 0; float newbd = 0; float newEnergy = 0;
        for(size_t i = 0;i < nRes; ++i)
            newEnergy += point->linearizeResidual(Hcalib, 1, residuals+i,newHdd, newbd, newIdepth, mSystemSettings);

        if(!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act)//setting_minIdepthH_act)
        {
            if (std::isfinite(lastEnergy) && mSystemSettings.LocalMapperConditionalDepthOptimization)
            {
                //Point was initialized but depth is bad!
                activateButDontOptimize = true;
                I3D_LOG(i3d::info) <<"activateButDontOptimize"<<point->getPixel2DHom().transpose();

                break;
            }
            if(print) printf("newHdd < 100 OptPoint: Not well-constrained (%zd res, H=%.1f). E=%f. SKIP!\n", nRes, newHdd,lastEnergy);
            I3D_LOG(i3d::info) <<"newHdd < 100 OptPoint: Not well-constrained"<<point->getPixel2DHom().transpose();

            return nullptr;
        }
        
        if(print) printf("%s %d (L %.2f) %s: %f -> %f (idepth %f)!\n",(true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT",iteration,log10(lambda),"",lastEnergy, newEnergy, newIdepth);

        if(newEnergy < lastEnergy)
        {
            currentIdepth = newIdepth;
            lastHdd = newHdd;
            lastbd = newbd;
            lastEnergy = newEnergy;
            for(size_t i = 0; i < nRes; ++i)
            {
                residuals[i].state_state = residuals[i].state_NewState;
                residuals[i].state_energy = residuals[i].state_NewEnergy;
            }

            lambda *= 0.5;
        }
        else
        {
            lambda *= 5;
        }

        if(std::abs(step) < 0.0001*currentIdepth)
            break;
    }
    if(!std::isfinite(currentIdepth))
    {
        LOG_THRESHOLD(i3d::info);
        printf("MAJOR ERROR! point idepth is nan after initialization (%f).\n", currentIdepth);
        I3D_LOG(i3d::info) <<"MAJOR ERROR! point idepth is nan after initialization"<<point->getPixel2DHom().transpose();

        return nullptr;		// yeah I'm like 99% sure this is OK on 32bit systems.
    }
    size_t numGoodRes{0};
    for(size_t i = 0; i < nRes; ++i)
        if(residuals[i].state_state == ResState::IN) numGoodRes++;

    if(static_cast<int>(numGoodRes) < minObs)
    {
        if(print) printf("OptPoint: OUTLIER!\n");
        return nullptr;
    }

    EdgePixel* p = new EdgePixel(*point);

    p->lastResiduals[0].first = nullptr;
    p->lastResiduals[0].second = ResState::OOB;
    p->lastResiduals[1].first = nullptr;
    p->lastResiduals[1].second = ResState::OOB;

    if (!mSystemSettings.LocalMapperOptimizeInitDepth || activateButDontOptimize)
    {
        currentIdepth = point->idepth;
    }
    p->setIdepthZero(currentIdepth);
    p->setIdepth(currentIdepth);
    p->setEdgeStatus(EdgePixel::ACTIVE);

    for(size_t i = 0; i < nRes; ++i)
        if(residuals[i].state_state == ResState::IN)
        {
            EdgeFrameResidual* r = new EdgeFrameResidual(p, p->host, residuals[i].target);
            if (!mSystemSettings.LocalMapperOptimizeInitDepth || 
                (activateButDontOptimize && mSystemSettings.LocalMapperConditionalDepthOptimization))
            {
                r->FLAG_DONT_OPT_DEPTH = true;
            }
            r->state_NewEnergy = r->state_energy = 0;
            r->state_NewState = ResState::OUTLIER;
            r->setState(ResState::IN);
            p->edgeResiduals.push_back(r);
            if(r->target == mFrameHessians.back())
            {
                p->lastResiduals[0].first = r;//.get();
                p->lastResiduals[0].second = ResState::IN;
            }
            else if(r->target == (mFrameHessians.size() < 2 ? 0 : mFrameHessians[mFrameHessians.size()-2]))//(allKeyFrameData.size() < 2 ? nullptr : allKeyFrameData[allKeyFrameData.size()-2].get()))
            {
                p->lastResiduals[1].first = r;//.get();
                p->lastResiduals[1].second = ResState::IN;
            }
        }

    if(print) printf("point activated!\n");
    LOG_THRESHOLD(i3d::info);
    return p;
}

void
LocalMapper::reAddConnectionsFromFrames(std::vector<std::unique_ptr<FrameData>>& allKeyFrames)
{
    
    for (const auto& fh : allKeyFrames)
        for (const auto& c : fh->getConstraints())
        {
            I3D_LOG(i3d::info) << "c.i " << c.i << "/ c.j: " << c.j;
            addFrameConstraintToGraph(c);
        }
}

void
LocalMapper::addFrameConstraintToGraph(const CeresConstraint& c) 
{
    addFrameConstraintToGraph(c.i,c.j);
}

void
LocalMapper::addFrameConstraintToGraph(const size_t frameId1, const size_t frameId2)
{
    mMargConnectivityGraph[(((uint64_t)frameId1) << 32) + ((uint64_t)frameId2)] = Vec2i(1,1);
    mMargConnectivityGraph[(((uint64_t)frameId2) << 32) + ((uint64_t)frameId1)] = Vec2i(1,1);
}

void
LocalMapper::publishGraph()
{
    for(auto* ow : mOutputWrapper)
        ow->publishMargGraph(mMargConnectivityGraph);
}

void
LocalMapper::resetLocalMapper(const size_t deleteTillFrameId)
{
    stopLocalMapping();
    I3D_LOG(i3d::info) << "resetLocalMapper: ";
    //clear the queue
    {
        std::lock_guard<std::mutex> l(mFrameHessiansQueueMutex);
        I3D_LOG(i3d::info) << "resetLocalMapper: ";
        //clear the queue.
        //NOTE: There's no clear for queue, thus we swap!
        std::queue<FrameData*>().swap(mFrameHessiansQueue);
    }
    //delete window
    {
        I3D_LOG(i3d::info) << "resetLocalMapper: ";
        mWindowedOptimizer.reset(new WindowedOptimizer(mSystemSettings));
        mWindowedOptimizer->setThreadReduce(&mThreadReduce);
        //Remove all the marginalized connections
        mMargConnectivityGraph.clear();
        mActiveResiduals.clear();
        std::lock_guard<std::mutex> l(mFrameHessiansMutex);
        //Before clearing the frame hessians, add connections
        //NOTE: We only implement the version, where we add the frames that are either in a loop or the fern db
        if (mSystemSettings.EnableLoopClosure)
        {
            size_t nConstraints{0};

            //go through 
            for (auto& frame : mFrameHessians)
            {
                I3D_LOG(i3d::info) << "frame: " << frame;
                //only for the frames that are not going to be deleted!
                if (frame->mFrameHeader->mFrameId < deleteTillFrameId)
                {
                    //There are two cases:
                    //in db or in loop -> keep
                    if (frame ->isInFernDbOrLoop())
                    {
                        //search in the window
                        for (const auto& fh : mFrameHessians)
                        {
                            if (frame != fh && fh->mFrameHeader->mFrameId < deleteTillFrameId && fh->isInFernDbOrLoop())
                            {   
                                const size_t diff = fh->idx < frame->idx ? frame->idx - fh->idx : fh->idx - frame->idx; //basically std::abs for size_t
                                if (frame->isInFernDbOrLoop() && fh->isInFernDbOrLoop() && diff <= mSystemSettings.LoopClosureNumberOfBAConstraints)// mSystemSettings.LoopClosureOnlyUseFernKF) || 
                                    // (diff <= mSystemSettings.LoopClosureNumberOfBAConstraints && !mSystemSettings.LoopClosureOnlyUseFernKF))
                                {
                                    frame->addConstraint(CeresConstraint(fh->getPRE_worldToCam()*frame->getPRE_camToWorld(),frame->mKeyFrameId,fh->mKeyFrameId));
                                    I3D_LOG(i3d::info) << "Adding constraint " << nConstraints << " for " << frame->mKeyFrameId << "->"<<fh->mKeyFrameId;
                                    ++nConstraints;
                                    addFrameConstraintToGraph(frame->mKeyFrameId,fh->mKeyFrameId);
                                }
                            }
                        }
                        if (nConstraints == 0)
                        {
                            I3D_LOG(i3d::info) << frame->mKeyFrameId << " not enough constraints!";
                            //We have a problem, since no frame in the window is in a loop or in the db
                            const FrameData* const newRefFrame = mParentMapper->findLastValidKeyframeNoLock(frame);
                            if (newRefFrame == frame) continue;
                            frame->addConstraint(CeresConstraint(newRefFrame->getPRE_worldToCam()*frame->getPRE_camToWorld(),frame->mKeyFrameId,newRefFrame->mKeyFrameId));
                            addFrameConstraintToGraph(frame->mKeyFrameId,newRefFrame->mKeyFrameId);
                            I3D_LOG(i3d::info) << frame->mKeyFrameId << " new constraints " << newRefFrame->mKeyFrameId
                                            << "pose: " << newRefFrame->getPRE_worldToCam().matrix3x4()
                                            << "pose: " << newRefFrame->mFrameHeader->getWorldToCam().matrix3x4();
                        }
                        else
                        {
                            I3D_LOG(i3d::info) << frame->mKeyFrameId << " with: " << nConstraints << " enough constraints!";
                        }
                    }
                    else //neither in db nor loop -> make to normal frame
                    {
                        //TODO: Replace by "mapper method!" update to new ref frame
                        const FrameData* const newRefFrame = mParentMapper->findLastValidKeyframeNoLock(frame);
                        I3D_LOG(i3d::info) << "Keyframe found: " << newRefFrame->mKeyFrameId << " allId: " << newRefFrame->mFrameHeader->mFrameId;
                        //TODO: Refactor to convert to normal frame!
                        const SE3Pose camToRef = newRefFrame->getPRE_worldToCam()*frame->getPRE_camToWorld(); //ok
                        I3D_LOG(i3d::info)  << "camToRef Before: " << frame->mFrameHeader->getCamToRef().matrix3x4()
                                            << " camToRef after: " << camToRef.matrix3x4()
                                            << " worldPose: " << frame->getPRE_camToWorld().matrix3x4()
                                            << " after: " << (newRefFrame->getPRE_camToWorld()*camToRef).matrix3x4()
                                            << " worldPose of new: " << newRefFrame->getPRE_camToWorld().matrix3x4();
                        //Make it to a "normal" frame
                        frame->mFrameHeader->setRefFrame(newRefFrame);
                        frame->mFrameHeader->setCamToRef(camToRef);
                        frame->mFrameHeader->updateCamToWorld(newRefFrame->getPRE_camToWorld()*camToRef);
                        I3D_LOG(i3d::info) << "camToRef after: " << frame->mFrameHeader->getCamToRef().matrix3x4(); 
                        frame->mFrameHeader->mIsKeyFrame = false;
                        mParentMapper->changeRefFrame(frame,newRefFrame);
                    }
                }
            }
        }
        mFrameHessians.clear();
    }   
}
}
