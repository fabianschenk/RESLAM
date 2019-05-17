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
#include "../config/Defines.h"
#include "boost/thread.hpp"
#include <stdio.h>
#include <iostream>

#include "Logging.h"

namespace RESLAM
{

template<typename Running>
class IndexThreadReduce
{

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	inline IndexThreadReduce()
	{
		nextIndex = 0;
		maxIndex = 0;
		stepSize = 1;
		callPerIndex = boost::bind(&IndexThreadReduce::callPerIndexDefault, this, _1, _2, _3, _4);

		running = true;
		for(size_t i=0;i<NumThreads;i++)
		{
			isDone[i] = false;
			gotOne[i] = true;
			workerThreads[i] = boost::thread(&IndexThreadReduce::workerLoop, this, i);
		}

	}
	inline ~IndexThreadReduce()
	{
		running = false;

		exMutex.lock();
		todo_signal.notify_all();
		exMutex.unlock();

		for(size_t i=0;i<NumThreads;i++)
			workerThreads[i].join();

	}

	inline void reduce(boost::function<void(int,int,Running*,int)> callPerIndex, int first, int end, int stepSize = 0)
	{
		memset(&stats, 0, sizeof(Running));

		if(stepSize == 0)
			stepSize = ((end-first)+NumThreads-1)/NumThreads;

		boost::unique_lock<boost::mutex> lock(exMutex);

		// save
		this->callPerIndex = callPerIndex;
		nextIndex = first;
		maxIndex = end;
		this->stepSize = stepSize;

		// go worker threads!
		for(size_t i=0;i<NumThreads;i++)
		{
			isDone[i] = false;
			gotOne[i] = false;
		}

		// let them start!
		todo_signal.notify_all();

		// wait for all worker threads to signal they are done.
		while(true)
		{
			// wait for at least one to finish
			done_signal.wait(lock);

			// check if actually all are finished.
			bool allDone = true;
			for(size_t i=0;i<NumThreads;i++)
				allDone = allDone && isDone[i];

			// all are finished! exit.
			if(allDone)
				break;
		}

		nextIndex = 0;
		maxIndex = 0;
		this->callPerIndex = boost::bind(&IndexThreadReduce::callPerIndexDefault, this, _1, _2, _3, _4);
	}

	Running stats;

private:
	boost::thread workerThreads[NumThreads];
	bool isDone[NumThreads];
	bool gotOne[NumThreads];

	boost::mutex exMutex;
	boost::condition_variable todo_signal;
	boost::condition_variable done_signal;

	int nextIndex;
	int maxIndex;
	int stepSize;

	bool running;

	boost::function<void(int,int,Running*,int)> callPerIndex;

	void callPerIndexDefault(int i, int j,Running* k, int tid)
	{
		printf("ERROR: should never be called....\n");
		assert(false);
	}

	void workerLoop(int idx)
	{
		boost::unique_lock<boost::mutex> lock(exMutex);

		while(running)
		{
			// try to get something to do.
			int todo = 0;
			bool gotSomething = false;
			if(nextIndex < maxIndex)
			{
				// got something!
				todo = nextIndex;
				nextIndex+=stepSize;
				gotSomething = true;
			}

			// if got something: do it (unlock in the meantime)
			if(gotSomething)
			{
				lock.unlock();

				assert(callPerIndex != 0);

				Running s; 
				memset(&s, 0, sizeof(Running));
				callPerIndex(todo, std::min(todo+stepSize, maxIndex), &s, idx);
				gotOne[idx] = true;
				lock.lock();
				stats += s;
			}
			// otherwise wait on signal, releasing lock in the meantime.
			else
			{
				if(!gotOne[idx])
				{
					lock.unlock();
					assert(callPerIndex != 0);
					Running s; memset(&s, 0, sizeof(Running));
					callPerIndex(0, 0, &s, idx);
					gotOne[idx] = true;
					lock.lock();
					stats += s;
				}
				isDone[idx] = true;
				done_signal.notify_all();
				todo_signal.wait(lock);
			}
		}
	}
};
}
