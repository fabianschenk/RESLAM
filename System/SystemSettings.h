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

#include "../config/Defines.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
namespace RESLAM {

struct SystemSettings
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    //Run with separate files
    SystemSettings(const std::string& settingsFileName,
                   const std::string& sensorSettingsFileName);

    void readSettings(const std::string& settingsFileName);
    bool readSensorSettings(const std::string& sensorSettingsFileName);
    void readViewerSettings(const std::string &settingsFileName);

    size_t readSizeT(const cv::FileStorage& fs, const std::string& str, const size_t defaultValue) const;

    size_t ViewerPointSize;
    bool SystemMultiThreading;
    //Settings for Tracker
    size_t TrackerEvalLvlForInit;
    bool TrackerSkipResdiualsOnLowerLvls;
    //if true, we try to register the frame onto the keyframe (REVO, IROS/BMVC)
    //if false, we try to register the kf onto the frame
    bool TrackerTrackFromFrameToKf;
    bool TrackerUseWeightsResidualsForErrorComputation = true;
    float TrackerResidualHuberWeight;
    bool TrackerUseEdgeWeight;
    bool TrackerUseEdgeFilter;
    bool TrackerShowIterationsDebug;
    bool TrackerShowInitDebug;
    float TrackerHistDist;
    Vec4f TrackerHistVotingWeights;

    /// Optical flow criteria is = sqrt(optFlowT/nGood)*opticalFlowTFactor + sqrt(optFlowRt/nGood)*opticalFlowRTFactor
    float TrackerOpticalFlowTFactor;  ///< Factor with which the optical flow with trans. is multiplied
    float TrackerOpticalFlowRTFactor; ///< Factor with which the optical flow with rot. and trans. is multiplied
    float TrackerOpticalFlowThreshold; ///< Threshold for optical flow constraint. If below ok else new keyframe
    float TrackerAvgResidualBeforeTrackingLoss; ///< If the avg. residual is above this threshold, we are probably lost

    //Maybe move those to the fixed settings of the tracker
    std::array<float,PyramidLevels> TrackerEdgeDistanceFilter={{30.0f,20.0f,10.0f}};
    std::array<size_t,PyramidLevels> TrackerMaxItsPerLvl={{100,100,100}};
    //lower for faster convergence
    std::array<float,PyramidLevels> TrackerConvergenceEps={{0.999f,0.999f,0.999f}};
    //higher for faster convergence
    std::array<float,PyramidLevels> TrackerStepSizeMin={{1e-16,1e-16,1e-16}};

    float TrackerLambdaSuccessFac = 0.5f;
    float TrackerLambdaFailFac = 2.0f;
    //Sensor settings
    std::string InputDatasetFile; //eg. associate file
    std::string InputDatasetFolder; //folder, where rgb/depth folders are
    int InputFormat;
    int InputSensorType;
    size_t InputSkipFirstNFrames;
    int InputStopAfterNFrames;
    size_t InputReadNFrames;
    float InputDepthScaleFactor;
    float InputDepthMin, InputDepthMax;
    bool InputSmoothEdgeImage;
    bool InputColorFormatRGB;
    bool InputComputeGradientsForEdgeDetector;
    double InputCannyEdgeTh1,InputCannyEdgeTh2;
    bool InputReadGT;
    bool OutputRecordImages;
    bool OutputPoses;
    std::string OutputPoseFileFolder;
    std::string DatasetConfigFileName; ///< Where to look for the config file

    cv::Size2i IntrinsicsImgSize;
    float IntrinsicsCamFx,IntrinsicsCamFy,IntrinsicsCamCx,IntrinsicsCamCy;

    //LocalMapper stuff
    bool EnableLocalMapping; ///< Flag that enables local mapping (windowed BA)
    bool LocalMapperUseCoarseDistanceMap;
    bool LocalMapperOptimizeDepth; ///< Flag that triggers depth optimization
    bool LocalMapperOptimizeInitDepth; ///< Flag that triggers initial depth optimization
    bool LocalMapperConditionalDepthOptimization; ///< Flag that triggers depth optimization
    bool LocalMapperAdaptMinActDist; ///< Flag that enables min act dist adaptation
    float LocalMapperMaxDistForValidPixels; ///< Maximum distance in the distance transform
    float LocalMapperMinDistInDistMapForValidPixels; ///< Minimum distance to other activated points
    bool LocalMapperShowResidualsInImage;
    float LocalMapperEdgeDistance; ///< Edge distance for Local Mapper
    bool LocalMapperLinearProcessing; ///< starts a reader thread!
    bool LocalMapperDoMarginalize; ///< flag for marginalization
    bool LocalMapperUseNewPoseUpdate;
    bool SystemDebugTextOutput; ///< sets LOG_THRESHOLD(i3d::debug)

    bool EnableLoopClosure; ///< Flag to enable loop closure
    size_t LoopClosureFramesToCheck;
    size_t LoopClosureMinDistBetweenFrames; ///< This sets the minimum distance between two candidates, where we don't close a loop
    size_t LoopClosureEdgeWeight;
    size_t LoopClosureNumberOfBAConstraints;
    size_t LoopClosureDoNotCloseLoopsIfClosedLastN; ///< Number of min dist between loop closures
    bool LoopClosureUseBAConstraints;
    bool LoopClosureLinearProcessing;
    bool LoopClosureOnlyUseFernKF;
    bool LoopClosureFixWindowPoses;
    bool LoopClosureAddAdjacentConstraints;
    bool LoopClosureDoBackCheck;
    float LoopClosureKeyframeThreshold;
    bool LoopClosureDebugShowTransformationImages;
    bool LoopClosureDebug;
    bool LoopClosureTrackerShowIterations;
    bool EnableRelocaliser; ///< Flag to enable Relocaliser
    Vec4f LoopClosureHistVotingWeights; ///< Weights that are used for the histogram voting to verify loop closures
    bool FernDatabaseAlwaysAddKf; ///< Flag 

    size_t RemoveNFramesAfterTrackLoss;         ///< Remove N frames if tracking was lost
    size_t RelocaliserMinNumberOfKeyframes;     ///< Minimum map size for relocalisation
};

}

