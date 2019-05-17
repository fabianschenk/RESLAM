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
#include <chrono>
#include <utility>
using namespace std::chrono;
class Timer
{
public:
    Timer(){}
    
    static inline auto getTime()
    {
        return high_resolution_clock::now();
    }

    //returns the time difference in microseconds
    static inline auto getTimeDiffMiS(const high_resolution_clock::time_point& start, const high_resolution_clock::time_point& end)
    {
        const auto range = (end < start ? start - end : end - start);
        return duration_cast<microseconds>(range).count(); 
    }
    
    //returns the time difference in milliseconds
    static inline auto getTimeDiffMs(const high_resolution_clock::time_point& start, const high_resolution_clock::time_point& end)
    {
        const auto range = (end < start ? start - end : end - start);
        return duration_cast<milliseconds>(range).count(); 

    }
};

