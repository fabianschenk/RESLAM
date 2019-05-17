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

#include "IndexThreadReduce.h"
#include "MatrixAccumulators.h"


#include "../config/Defines.h"
#include <math.h>
#include <vector>

namespace RESLAM
{

class EFPoint;
class WindowedOptimizer;


class AccumulatedSCHessianSSE
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	inline AccumulatedSCHessianSSE()
	{
		for(size_t i = 0; i < NumThreads; i++)
		{
			accE[i] = nullptr;
			accEB[i] = nullptr;
			accD[i] = nullptr;
			nframes[i] = 0;
		}
    }
	inline ~AccumulatedSCHessianSSE()
	{
		for(size_t i=0;i<NumThreads;i++)
		{
			if(accE[i] != nullptr) delete[] accE[i];
			if(accEB[i] != nullptr) delete[] accEB[i];
			if(accD[i] != nullptr) delete[] accD[i];
		}
    }

	inline void setZero(int n, int min=0, int max=1, Vec10* stats=0, int tid=0)
	{
		if(n != nframes[tid])
		{
			if(accE[tid] != nullptr) delete[] accE[tid];
			if(accEB[tid] != nullptr) delete[] accEB[tid];
			if(accD[tid] != nullptr) delete[] accD[tid];
            accE[tid] = new AccumulatorXX<HessianSize,CPARS>[n*n];
            accEB[tid] = new AccumulatorX<HessianSize>[n*n];
            accD[tid] = new AccumulatorXX<HessianSize,HessianSize>[n*n*n];
		}



		accbc[tid].initialize();
		accHcc[tid].initialize();

		for(int i=0;i<n*n;i++)
		{
			accE[tid][i].initialize();
			accEB[tid][i].initialize();

			for(int j=0;j<n;j++)
				accD[tid][i*n+j].initialize();
		}
		nframes[tid]=n;
	}
    void stitchDouble(MatXX &H_sc, VecX &b_sc, WindowedOptimizer const * const EF, int tid=0);
	void addPoint(EFPoint* p, bool shiftPriorToZero, int tid=0);


    void stitchDoubleMT(IndexThreadReduce<Vec10>* red, MatXX &H, VecX &b, WindowedOptimizer const * const EF, bool MT)
	{
		// sum up, splitting by bock in square.
        if(MT)
        {
            MatXX Hs[NumThreads];
            VecX bs[NumThreads];
            for(size_t i=0; i < NumThreads; i++)
            {
                assert(nframes[0] == nframes[i]);
                Hs[i] = MatXX::Zero(nframes[0]*HessianSize+CPARS, nframes[0]*HessianSize+CPARS);
                bs[i] = VecX::Zero(nframes[0]*HessianSize+CPARS);
            }

            red->reduce(boost::bind(&AccumulatedSCHessianSSE::stitchDoubleInternal,
                this,Hs, bs, EF,  _1, _2, _3, _4), 0, nframes[0]*nframes[0], 0);

            // sum up results
            H = Hs[0];
            b = bs[0];

            for(size_t i = 1; i < NumThreads; i++)
            {
                H.noalias() += Hs[i];
                b.noalias() += bs[i];
            }
        }
        else
		{
            H = MatXX::Zero(nframes[0]*HessianSize+CPARS, nframes[0]*HessianSize+CPARS);
            b = VecX::Zero(nframes[0]*HessianSize+CPARS);
			stitchDoubleInternal(&H, &b, EF,0,nframes[0]*nframes[0],0,-1);
		}

		// make diagonal by copying over parts.
		for(int h=0;h<nframes[0];h++)
		{
            int hIdx = CPARS+h*HessianSize;
            H.block<CPARS,HessianSize>(0,hIdx).noalias() = H.block<HessianSize,CPARS>(hIdx,0).transpose();
		}
	}


    AccumulatorXX<HessianSize,CPARS>* accE[NumThreads];
    AccumulatorX<HessianSize>* accEB[NumThreads];
    AccumulatorXX<HessianSize,HessianSize>* accD[NumThreads];
	AccumulatorXX<CPARS,CPARS> accHcc[NumThreads];
	AccumulatorX<CPARS> accbc[NumThreads];
	int nframes[NumThreads];


	void addPointsInternal(
			std::vector<EFPoint*>* points, bool shiftPriorToZero,
			int min=0, int max=1, Vec10* stats=0, int tid=0)
	{
		for(int i=min;i<max;i++) addPoint((*points)[i],shiftPriorToZero,tid);
	}

private:

	void stitchDoubleInternal(
            MatXX* H, VecX* b, WindowedOptimizer const * const EF,
			int min, int max, Vec10* stats, int tid);
};

}

