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
#include <memory>
#include <Eigen/Eigen>

#include <thread>

#include "System/System.h"
#include "Utils/Logging.h"
#ifdef WITH_PANGOLIN_VIEWER
#include "GUI/Pangolin/PangolinDSOViewer.h"
#endif


int main(int argc, char ** argv)
{
    LOG_THRESHOLD(i3d::info);
    if (argc < 2) //first is name of program
    {
        I3D_LOG(i3d::error) << "Not enough input arguments: RESLAM configFile.yaml datasetFile.yaml. Dataset file is optional!";
        exit(0);
    }
    
    //convert argument to string
    const std::string settingsFile = argv[1],
                      datasetFile = (argc > 2 ? argv[2] : "");
    int nRuns = 0;
    std::vector<RESLAM::IOWrap::Output3DWrapper*> outputWrappers;
    //now start the system
    while (true)
    {
        RESLAM::System reSlam(settingsFile,datasetFile);//,datasetFile,nRuns);

#ifdef WITH_PANGOLIN_VIEWER
        auto viewer = std::make_unique<RESLAM::IOWrap::PangolinDSOViewer>(640,480, reSlam.returnSystemSettings(),false);
        outputWrappers.push_back(viewer.get());
        reSlam.addOutputWrapper(outputWrappers);
        std::thread runthread([&]() {
            if (viewer != nullptr) viewer->run();
            reSlam.stopSystem();
        });
#endif
        if (!reSlam.startSystem())
        {
#ifdef WITH_PANGOLIN_VIEWER
            runthread.join();
#endif
            I3D_LOG(i3d::info) << "Finished all datasets!";
            return EXIT_SUCCESS;
        }
#ifdef WITH_PANGOLIN_VIEWER        
        for (auto& ow : reSlam.getOutputWrapper())
        {
            I3D_LOG(i3d::info) << "Waiting for threads to join";
            ow->join();
            I3D_LOG(i3d::info) << "One thread joined!";// for threads to join";
        }
        runthread.join();
#endif
        nRuns++;
        I3D_LOG(i3d::info) << "nRuns" << nRuns << "runthread.join()";
        break;
    }
    I3D_LOG(i3d::info) << "Thanks for using RESLAM!";
    return EXIT_SUCCESS;
}
