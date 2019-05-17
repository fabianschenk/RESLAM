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
#include <sophus/se3.hpp>
#include <memory>
#include <map>
namespace RESLAM
{
class EdgePixel;
class FrameData;
struct DetectedEdge;

using PREC = double; //Computing Precision
constexpr auto DOUBLE_INF = std::numeric_limits<double>::infinity();
// using PREC_EVAL = float;
constexpr auto FLOAT_INF = std::numeric_limits<float>::infinity();
using POSEPREC = double; //Pose Precision
using SE3Pose = Sophus::SE3<POSEPREC>;
using SO3Pose = Sophus::SO3<POSEPREC>;
using KeyframePoseVector = std::vector<std::pair<FrameData*,SE3Pose>,Eigen::aligned_allocator<SE3Pose>>;
struct CeresConstraint
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CeresConstraint(const SE3Pose& T_ji, size_t i_, size_t j_):T_j_i(T_ji),i(i_),j(j_){}
    SE3Pose T_j_i; //transformation from i to j
    size_t i, j; //keyframeId
};

struct CeresPose
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CeresPose(const SE3Pose& pose)
    {
        t = pose.translation();
        q = Eigen::Quaterniond(pose.unit_quaternion());
    }
    SE3Pose returnPose() const { return SE3Pose(q,t); }
    Eigen::Vector3d t; ///< translation
    Eigen::Quaterniond q; ///< quaternion
    bool flagFixPose = false;
    void setPoseFixed() { flagFixPose = true; }
};
using PoseVector = std::vector<SE3Pose,Eigen::aligned_allocator<SE3Pose>>;
using CeresPoseVector = std::vector<CeresPose,Eigen::aligned_allocator<CeresPose>>;
using RelPoseConstraints = std::vector<CeresConstraint, Eigen::aligned_allocator<CeresConstraint>>;

constexpr size_t PyramidLevels{3};
constexpr size_t NUMBER_OF_VIEWER_COLORS{3};
constexpr size_t NumThreads{6};
constexpr bool setting_debugout_runquiet{true};


constexpr PREC SCALE_IDEPTH{1.0};		// scales internal value to idepth.
constexpr PREC SCALE_XI_ROT{1.0};
constexpr PREC SCALE_XI_TRANS{0.5};
constexpr PREC SCALE_F{1.0}; //50.0f //50.0f //NO idea
constexpr PREC SCALE_C{1.0}; //50.0f //50.0f //NO idea
constexpr PREC SCALE_W{1.0};
//#define SCALE_A 10.0f
//#define SCALE_B 1000.0f

constexpr PREC SCALE_IDEPTH_INVERSE{1.0/SCALE_IDEPTH};
constexpr PREC SCALE_XI_ROT_INVERSE {1.0/SCALE_XI_ROT};
constexpr PREC SCALE_XI_TRANS_INVERSE{1.0/SCALE_XI_TRANS};
constexpr PREC SCALE_F_INVERSE{1.0/SCALE_F};
constexpr PREC SCALE_C_INVERSE{1.0/SCALE_C};
constexpr PREC SCALE_W_INVERSE{1.0/SCALE_W};


constexpr size_t setting_minGoodActiveResForMarg{3};
constexpr size_t setting_minGoodResForMarg{4};
constexpr float setting_minIdepthH_act{100};
constexpr float setting_minIdepthH_marg{50};
constexpr bool setting_forceAcceptStep{true};

constexpr int   setting_maxOptIterations=6; // max GN iterations.
constexpr int   setting_minOptIterations=1; // min GN iterations.
constexpr float setting_thOptIterations=1.2; // factor on break threshold for GN iteration (larger = break earlier)

constexpr float setting_minGradHistAdd = 7;
constexpr float setting_kfGlobalWeight = 1;   // general weight on threshold, the larger the more KF's are taken (e.g., 2 = double the amount of KF's).

constexpr float setting_idepthFixPriorMargFac = 600*600;
constexpr int SOLVER_SVD{1};
constexpr int SOLVER_ORTHOGONALIZE_SYSTEM{2};
constexpr int SOLVER_ORTHOGONALIZE_POINTMARG{4};
constexpr int SOLVER_ORTHOGONALIZE_FULL{8};
constexpr int SOLVER_SVD_CUT7{16};
constexpr int SOLVER_REMOVE_POSEPRIOR{32};
constexpr int SOLVER_USE_GN{64};
constexpr int SOLVER_FIX_LAMBDA{128};
constexpr int SOLVER_ORTHOGONALIZE_X{256};
constexpr int SOLVER_MOMENTUM{512};
constexpr int SOLVER_STEPMOMENTUM{1024};
constexpr int SOLVER_ORTHOGONALIZE_X_LATER{2048};
constexpr int setting_solverMode = SOLVER_FIX_LAMBDA | SOLVER_ORTHOGONALIZE_X_LATER;
constexpr double setting_solverModeDelta = 0.00001;
constexpr float setting_margWeightFac{0.5*0.5};
constexpr float setting_initialCameraMatrix = 5e9;

constexpr float setting_desiredPointDensity = 2000; // aimed total points in the active window.
//Defines
using Vec4 = Eigen::Matrix<PREC,4,1>;
using Vec3 = Eigen::Matrix<PREC,3,1>;
using Vec2 = Eigen::Matrix<PREC,2,1>;
using Vec2i = Eigen::Matrix<int,2,1>;
using Mat33 = Eigen::Matrix<PREC,3,3>;

using DetectedEdgesPtrVec = std::vector<DetectedEdge*>;
using DetectedEdgesUniquePtrVec = std::vector<std::unique_ptr<DetectedEdge>>;
using EdgePixelVec = std::vector<EdgePixel*>;
using ConnectivityMap = std::map<uint64_t,Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<uint64_t, Eigen::Vector2i> > >;// &connectivity

//Loop Closer
constexpr float FernDBKfHarvestingThreshold{0.2};
constexpr size_t FernDBNumFerns = 500;
constexpr size_t FernDBNumDecisionsionsPerFern = 4;
constexpr size_t FernDBNumCodes = (1 << FernDBNumDecisionsionsPerFern);
constexpr int FernDBNumOfChannels{3};
constexpr int FernDBNumOfDecisions = FernDBNumDecisionsionsPerFern / FernDBNumOfChannels;
constexpr size_t NumOfConnectionsFromMarginalization{3}; ///< If a frame is marginalized, keep at least N opt. transformations!
//Loop Closer End


//DSO DEFINES
constexpr size_t HessianSize{6};
constexpr size_t MOTION_DOF{6};
//camera parameters
constexpr int CPARS{4};
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatXX;
typedef Eigen::Matrix<double,CPARS,CPARS> MatCC;
#define MatToDynamic(x) MatXX(x)
typedef Eigen::Matrix<double,CPARS,10> MatC10;
typedef Eigen::Matrix<double,10,10> Mat1010;
typedef Eigen::Matrix<double,13,13> Mat1313;

typedef Eigen::Matrix<double,8,10> Mat810;
typedef Eigen::Matrix<double,8,3> Mat83;
typedef Eigen::Matrix<double,6,6> Mat66;
typedef Eigen::Matrix<double,5,3> Mat53;
typedef Eigen::Matrix<double,4,3> Mat43;
typedef Eigen::Matrix<double,4,2> Mat42;
typedef Eigen::Matrix<double,3,3> Mat33;
typedef Eigen::Matrix<double,2,2> Mat22;
typedef Eigen::Matrix<double,HessianSize,CPARS> Mat6C;
typedef Eigen::Matrix<double,CPARS,HessianSize> MatC6;
typedef Eigen::Matrix<float,HessianSize,CPARS> Mat6Cf;
typedef Eigen::Matrix<float,CPARS,HessianSize> MatC6f;
typedef Eigen::Matrix<double,7,7> Mat77;
typedef Eigen::Matrix<double,CPARS,1> VecC;
typedef Eigen::Matrix<float,CPARS,1> VecCf;
typedef Eigen::Matrix<double,13,1> Vec13;
typedef Eigen::Matrix<double,10,1> Vec10;
typedef Eigen::Matrix<double,9,1> Vec9;
typedef Eigen::Matrix<double,7,1> Vec7;
typedef Eigen::Matrix<double,6,1> Vec6;
typedef Eigen::Matrix<double,5,1> Vec5;
typedef Eigen::Matrix<double,4,1> Vec4;
typedef Eigen::Matrix<double,3,1> Vec3;
typedef Eigen::Matrix<double,2,1> Vec2;
typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecX;
typedef Eigen::Matrix<float,3,3> Mat33f;
typedef Eigen::Matrix<float,10,3> Mat103f;
typedef Eigen::Matrix<float,2,2> Mat22f;
typedef Eigen::Matrix<float,3,1> Vec3f;
typedef Eigen::Matrix<float,2,1> Vec2f;
typedef Eigen::Matrix<float,6,1> Vec6f;

typedef Eigen::Matrix<double,4,9> Mat49;
typedef Eigen::Matrix<double,8,9> Mat89;

typedef Eigen::Matrix<double,9,4> Mat94;
typedef Eigen::Matrix<double,9,8> Mat98;

typedef Eigen::Matrix<double,8,1> Mat81;
typedef Eigen::Matrix<double,1,8> Mat18;
typedef Eigen::Matrix<double,9,1> Mat91;
typedef Eigen::Matrix<double,1,9> Mat19;


typedef Eigen::Matrix<double,8,4> Mat84;
typedef Eigen::Matrix<double,4,8> Mat48;
typedef Eigen::Matrix<double,4,4> Mat44;


typedef float VecNRf;
typedef Eigen::Matrix<float,12,1> Vec12f;
typedef Eigen::Matrix<float,1,6> Mat16f;
typedef Eigen::Matrix<float,6,6> Mat66f;
typedef Mat66f HessianTypef;
typedef Mat66 HessianType;
typedef Eigen::Matrix<float,10,1> Vec10f;
typedef Eigen::Matrix<float,6,6> Mat66f;
typedef Eigen::Matrix<float,4,1> Vec4f;
typedef Eigen::Matrix<float,4,4> Mat44f;
typedef Eigen::Matrix<float,12,12> Mat1212f;
typedef Eigen::Matrix<float,12,1> Vec12f;

typedef Eigen::Matrix<float,13,1> Vec13f;
typedef Eigen::Matrix<float,9,9> Mat99f;
typedef Eigen::Matrix<float,9,1> Vec9f;

typedef Eigen::Matrix<float,4,2> Mat42f;
typedef Eigen::Matrix<float,6,2> Mat62f;
typedef Eigen::Matrix<float,1,2> Mat12f;

typedef Eigen::Matrix<float,Eigen::Dynamic,1> VecXf;
typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> MatXXf;
constexpr size_t ACCUMULATOR_SIZE{11}; //6 motion, 4, camera, 1 for b vector
typedef Eigen::Matrix<float,11,11> Mat1111f;
typedef Eigen::Matrix<double,ACCUMULATOR_SIZE,ACCUMULATOR_SIZE> MatPCPC;
typedef Eigen::Matrix<float,ACCUMULATOR_SIZE,ACCUMULATOR_SIZE> MatPCPCf;
typedef Eigen::Matrix<double,ACCUMULATOR_SIZE,1> VecPC;
typedef Eigen::Matrix<float,ACCUMULATOR_SIZE,1> VecPCf;

typedef Eigen::Matrix<float,14,14> Mat1414f;
typedef Eigen::Matrix<float,14,1> Vec14f;
typedef Eigen::Matrix<double,14,14> Mat1414;
typedef Eigen::Matrix<double,14,1> Vec14;

//DSO END

struct RawResidualJacobian
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // ================== new structure: save independently =============.
    float resF;

    //see Eq. (13) DSO paper
    // the two rows of d[x,y]/d[xi].
    // motion derivative
    Vec6f Jpdxi[2];			// 2x6

    // the two rows of d[x,y]/d[C].
    //camera parameters
    VecCf Jpdc[2];			// 2x4

    //inverse depth
    // the two rows of d[x,y]/d[idepth].
    Vec2f Jpdd;				// 2x1

    // the two columns of d[r]/d[x,y].
    //since we only have one residual, this is essentially float [2]
    Vec2f JIdx;

    //we will skip Jabf at the moment!
    // = the two columns of d[r] / d[ab]

    // = JIdx^T * JIdx (inner product). Only as a shorthand.
    Mat22f JIdx2;				// 2x2
    // = Jab^T * JIdx (inner product). Only as a shorthand.
    //Eigen::Matrix2f JabJIdx;			// 2x2
    // = Jab^T * Jab (inner product). Only as a shorthand.
    //Eigen::Matrix2f Jab2;			// 2x2

};
}
