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
#pragma once
namespace RESLAM
{
    
class Mapper;
class Tracker;
class FrameData;
class SystemSettings;
namespace FernRelocLib
{
    class FernRelocaliser;
}
class Relocaliser
{
    public:
        Relocaliser(const SystemSettings& config, FernRelocLib::FernRelocaliser* fernRelocaliser, Mapper* mapper, Tracker* tracker);
        int relocalise(FrameData* currFrame) const;
    private:
        //Fern relocaliser
        const SystemSettings& mSystemSettings;
        FernRelocLib::FernRelocaliser* mFernRelocaliser; ///< Pointer to FernRelocaliser
        Mapper* mGlobalMapperPtr;  ///< Pointer to global mapper
        Tracker* mTrackerPtr; ///< Tracker that used for loop estimation and relocalisation
};
}