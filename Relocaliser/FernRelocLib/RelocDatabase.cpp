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
 * Some of the code is also part of InfiniTAM
 * Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM
 * 
 */
#include "RelocDatabase.h"

#include <fstream>
#include <stdexcept>
#include "../../Utils/Logging.h"
#include "../../config/Defines.h"

namespace RESLAM
{
namespace FernRelocLib
{
RelocDatabase::RelocDatabase()
{
	mTotalEntries = 0;
	mCodeLength = FernDBNumFerns;
	mCodeFragmentDim = FernDBNumCodes;
	mIds = new std::vector<int>[FernDBNumFerns*FernDBNumCodes];
}

RelocDatabase::~RelocDatabase()
{
	delete[] mIds;
}

std::vector<float> 
RelocDatabase::computeDistancesToKeyframes(const char *codeFragments)//, int &similarFrameId, float &distances)
{
	std::vector<float> distances;
	if (mTotalEntries == 0) return distances;
	I3D_LOG(i3d::info) << "mTotalEntries " << mTotalEntries;
	std::vector<int> similarities(mTotalEntries); //everything should be zero?
	std::fill(similarities.begin(),similarities.end(),0);
	for (int f = 0; f < mCodeLength; f++)
	{
		if (codeFragments[f] < 0) continue;
		const std::vector<int> *sameCode = &(mIds[f * mCodeFragmentDim + codeFragments[f]]);

		//now go through all the "same" codes!
		I3D_LOG(i3d::detail) << "sameCode->size(): " << sameCode->size();
		for (unsigned int i = 0; i < sameCode->size(); ++i)
		{
			similarities[(*sameCode)[i]]++;
			I3D_LOG(i3d::detail) << "similarities[(*sameCode)[i]]: " << i << ": " << similarities[(*sameCode)[i]];
		}
	}
	
	std::transform(similarities.cbegin(),similarities.cend(),std::back_inserter(distances),
					[this](const auto& similarity){return 1.0-static_cast<float>(similarity)/static_cast<float>(mCodeLength);});
	return distances;
}

// returns ID of newly added entry
size_t 
RelocDatabase::addEntry(const char *codeFragments)
{
	const auto newId = mTotalEntries;
	mTotalEntries++;
	for (int f = 0; f < mCodeLength; f++)
	{
		if (codeFragments[f] < 0) continue;
		std::vector<int> *sameCode = &(mIds[f * mCodeFragmentDim + codeFragments[f]]);
		sameCode->push_back(newId);
	}

	return newId;
}

void
RelocDatabase::resetDatabase()
{
	delete[] mIds;
	mIds = new std::vector<int>[FernDBNumFerns*FernDBNumCodes];
	mTotalEntries = 0;
}

// returns ID of newly added entry
void 
RelocDatabase::removeEntry(const char *codeFragments, const size_t dbId)
{
	I3D_LOG(i3d::info) << "removeEntry: " << dbId;
	for (int f = 0; f < mCodeLength; f++)
	{
		if (codeFragments[f] < 0) continue;
		std::vector<int> *sameCode = &(mIds[f * mCodeFragmentDim + codeFragments[f]]);
		const size_t sizeBefore = sameCode->size();
		sameCode->erase(std::remove_if(sameCode->begin(),sameCode->end(),
							[dbId](const int id){ return id == static_cast<int>(dbId); }),sameCode->end());
		const size_t sizeAfter = sameCode->size();
		I3D_LOG(i3d::info) << "sizeBefore: " << sizeBefore << " after: " << sizeAfter;
	}
	mTotalEntries--;
}

//we assume that we always delete the last entry
bool RelocDatabase::deleteLastEntry(const char *codeFragments)
{
    for (int f = 0; f < mCodeLength; f++)
    {
        if (codeFragments[f] < 0) continue;
        std::vector<int> *sameCode = &(mIds[f * mCodeFragmentDim + codeFragments[f]]);
        I3D_LOG(i3d::detail) << f <<" before: " << sameCode->size();
        sameCode->pop_back();
        I3D_LOG(i3d::detail) << f <<" after: " << sameCode->size();
    }
    mTotalEntries--;
    return true;
}

void RelocDatabase::SaveToFile(const std::string &framesFileName) const
{
	std::ofstream ofs(framesFileName.c_str());
	if (!ofs) throw std::runtime_error("Could not open " + framesFileName + " for reading");

	ofs << mCodeLength << " " << mCodeFragmentDim << " " << mTotalEntries << "\n";
	int dimTotal = mCodeLength * mCodeFragmentDim;
	for (int i = 0; i < dimTotal; i++)
	{
		ofs << mIds[i].size() << " ";
		for (size_t j = 0; j < mIds[i].size(); j++) ofs << mIds[i][j] << " ";
		ofs << "\n";
	}
}

void RelocDatabase::LoadFromFile(const std::string &filename)
{
	std::ifstream ifs(filename.c_str());
	if (!ifs) throw std::runtime_error("unable to load " + filename);

	ifs >> mCodeLength >> mCodeFragmentDim >> mTotalEntries;
	int len = 0, id = 0, dimTotal = mCodeFragmentDim * mCodeLength;
	for (int i = 0; i < dimTotal; i++)
	{
		ifs >> len;
		std::vector<int> *sameCode = &(mIds[i]);
		for (int j = 0; j < len; j++)
		{
			ifs >> id;
			sameCode->push_back(id);
		}
	}
}
}
}