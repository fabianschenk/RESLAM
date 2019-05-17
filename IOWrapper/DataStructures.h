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
#include <stddef.h>
#include <memory>
#include <opencv2/core/mat.hpp>
#include "../config/Defines.h"
#include "../GUI/MinimalImage.h"

namespace RESLAM
{

class EdgePixel;
class EFResidual;
class EFPoint;
class FrameData;
class EFFrame;
class CoarseDistanceMap;
class WindowedOptimizer;
class SystemSettings;

class CameraMatrix
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    float fx[PyramidLevels],fy[PyramidLevels],cx[PyramidLevels],cy[PyramidLevels];
    float fx_inv[PyramidLevels],fy_inv[PyramidLevels];
    float cx_div_fx,cy_div_fy; //the ratio always stays the same!
    Mat33f K[PyramidLevels], K_inv[PyramidLevels];
    cv::Mat map1, map2;
    cv::Mat distCoeff;
    int wBounds[PyramidLevels], hBounds[PyramidLevels];
    Vec4f imageBounds[PyramidLevels];
    int width[PyramidLevels], height[PyramidLevels];
    int area[PyramidLevels];

    // normal mode: use the optimized parameters everywhere!
    inline auto fxl() const {return value_scaledf[0];}
    inline auto fyl() const {return value_scaledf[1];}
    inline auto cxl() const {return value_scaledf[2];}
    inline auto cyl() const {return value_scaledf[3];}
    inline auto fxli() const {return value_scaledi[0];}
    inline auto fyli() const {return value_scaledi[1];}
    inline auto cxli() const {return value_scaledi[2];}
    inline auto cyli() const {return value_scaledi[3];}
    
    //Optimization stuff
    VecC step;
    VecC step_backup;
    VecC value_backup;
    VecC value;
    VecC value_minus_value_zero;
    //not scaled in my case
    VecCf value_scaledf;
    VecCf value_scaledi;
    VecC value_zero;
    VecC value_scaled;

    CameraMatrix(const SystemSettings& settings);

    //todo setter and getter
    void setLvl0FromParameters(float fx_0, float fy_0, float cx_0, float cy_0, int width_0, int height_0);
    void setFromCamMat(const CameraMatrix& camM) { setLvl0FromParameters(camM.fx[0], camM.fy[0], camM.cx[0], camM.cy[0], camM.width[0], camM.height[0]); }
    void makeK();
    inline void setValueScaled(const VecC &value_scaled);
    inline void setValue(const VecC &value)
    {
        // [0-3: Kl, 4-7: Kr, 8-12: l2r]
        this->value = value;
        value_scaled[0] = SCALE_F * value[0];
        value_scaled[1] = SCALE_F * value[1];
        value_scaled[2] = SCALE_C * value[2];
        value_scaled[3] = SCALE_C * value[3];

        this->value_scaledf = this->value_scaled.cast<float>();
        this->value_scaledi[0] = 1.0f / this->value_scaledf[0];
        this->value_scaledi[1] = 1.0f / this->value_scaledf[1];
        this->value_scaledi[2] = - this->value_scaledf[2] / this->value_scaledf[0];
        this->value_scaledi[3] = - this->value_scaledf[3] / this->value_scaledf[1];
        this->value_minus_value_zero = this->value - this->value_zero;
    }

};

enum class ResLocation {ACTIVE=0, LINEARIZED, MARGINALIZED, NONE};
enum class ResState {IN=0, OOB, OUTLIER};

class FrameFramePrecalc
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // static values
    static int instanceCounter;
    FrameData* host;	// defines row
    FrameData* target;	// defines column

    // precalc values
    Mat33f PRE_RTll, PRE_KRKiTll, PRE_RKiTll, PRE_RTll_0;
    Vec3f PRE_tTll;
    Vec3f PRE_KtTll;
    Vec3f PRE_tTll_0;
    float distanceLL;
    inline ~FrameFramePrecalc() {}
    inline FrameFramePrecalc() {host = target = nullptr;}
    void set(FrameData* host, FrameData* target, const CameraMatrix& HCalib );
};

/**
 * Struct FrameSet stores all the images on the top lvl
 */
struct FrameSet
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    FrameSet() = delete;
    FrameSet(const cv::Mat& gray_, const cv::Mat& depth_, const cv::Mat& rgb_, double _timestamp, const SystemSettings &config);//: timestamp(_timestamp),depthJet(depth_.cols, depth_.rows)
    double mTimestamp;
    
    MinimalImageB3 depthJet;    ///< Depth image in jet colors on lvl0 
    cv::Mat depth;              ///< Depth image on lvl 0
    cv::Mat edgeImage;          ///< Edge image on lvl 0    
    cv::Mat grayScaleImg;       ///< Gray scale image on lvl 0
    cv::Mat grayScaleImgF;      ///< Gray scale image as float on lvl 0
    
    cv::Mat distanceTransform;  ///< Distance transform on lvl 0
    
    cv::Mat colorImg;
    cv::Mat gradMagn;
    DetectedEdgesUniquePtrVec mValidEdgePixels;
};

//all public
class FrameHeader
{
    public: 
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        double mTimestamp; //unique timestamp
        size_t mFrameId; //unique frame id
        bool mIsKeyFrame;
        FrameHeader(double timeStamp, size_t frameId):
            mTimestamp(timeStamp), mFrameId(frameId),mIsKeyFrame(false)
        {}
        void updateCamToWorld(const SE3Pose& T_world_cam)
        {
            mT_cam_world = T_world_cam.inverse();
            mT_world_cam = T_world_cam;
        }
        void updateWorldToCam(const SE3Pose& T_cam_world)
        {
            mT_cam_world = T_cam_world;
            mT_world_cam = T_cam_world.inverse();
        }

        void setRefToCam(const SE3Pose& T_cam_ref)
        {
            mT_ref_cam = T_cam_ref.inverse();
            mT_cam_ref = T_cam_ref;
        }
        void setCamToRef(const SE3Pose& T_ref_cam)
        {
            mT_cam_ref = T_ref_cam.inverse();
            mT_ref_cam = T_ref_cam;
        }
        void makeKeyFrame()
        {
            mIsKeyFrame = true;
            mT_ref_cam = SE3Pose();
            mT_cam_ref = SE3Pose();
        }

        SE3Pose getCamToWorld() const { return mT_world_cam; }
        SE3Pose getWorldToCam() const { return mT_cam_world; }
        SE3Pose getRefToCam() const { return mT_cam_ref; }
        SE3Pose getCamToRef() const { return mT_ref_cam; }
        void setRefFrame(const FrameData * const refFrame) { mReferenceFrame = refFrame; }
        const FrameData* getRefFrame() const { return mReferenceFrame; }
        const SE3Pose getWorldPoseFromRef() const;
        
        int marginalizedAt = -1;
        double movedByOpt;
    private:
        SE3Pose mT_cam_world; //transforms a point from the world into the camera
        //This is basically also the pose
        SE3Pose mT_world_cam; //transforms a point from the camera into the world

        //This will be set to Identiy and this if it's a keyframe
        SE3Pose mT_ref_cam; //transforms a point from the camera into its reference frame
        SE3Pose mT_cam_ref; //transforms a point from its reference frame into the camera
        const FrameData* mReferenceFrame;

        
};

class FrameData
{   
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using FrameFramePrecalcVec = std::vector<FrameFramePrecalc,Eigen::aligned_allocator<FrameFramePrecalc>>;
    FrameHeader* mFrameHeader;
    FrameData(FrameHeader* frameHeader, std::unique_ptr<FrameSet> frameSet, const SystemSettings& config):
        mFrameHeader(frameHeader), mFrameSet(std::move(frameSet)), mOptStructureAlreadyComputed(false), mSystemSettings(config)
    {
        mFernCode = nullptr;
        reprojDebug = mFrameSet->colorImg.clone();
    }
    ~FrameData();
    void makeKeyFrame(const CameraMatrix& camMat);
    void prepareForTracking(const SystemSettings& settings, const CameraMatrix& camMat);
    const Vec3f *getOptStructurePtr(const size_t lvl) const;
    const auto& getDetectedEdges() const { return mFrameSet->mValidEdgePixels;}
    const auto& getEdgesGood() const { return mEdgesGood;}
    const auto& getEdgesOut() const { return mEdgesOut;}
    const auto& getEdgesMarginalized() const { return mEdgesMarginalized;}
    //These two poses are used for the LocalMapper!
    size_t mKeyFrameId; // incremental ID for keyframes only!
    std::unique_ptr<FrameSet> mFrameSet; //We take the ownership
    bool mFlaggedForMarginalization = false;
    void makeTargetPrecalc();
    double computeTargetPrecalcDistScore(const FrameData * const latest) const;
    void computeTargetPrecalc(const std::vector<FrameData*>& localWindow, const CameraMatrix& camMat);

    EIGEN_STRONG_INLINE const SE3Pose &get_worldToCam_evalPT() const {return worldToCam_evalPT;}
    EIGEN_STRONG_INLINE const Vec6 &get_state_zero() const {return state_zero;}
    EIGEN_STRONG_INLINE const Vec6 &get_state() const {return state;}
    EIGEN_STRONG_INLINE const Vec6 &get_state_scaled() const {return state_scaled;}
    EIGEN_STRONG_INLINE const Vec6 get_state_minus_stateZero() const {return get_state() - get_state_zero();}

    const SE3Pose& getPRE_camToWorld() const {return PRE_camToWorld;}
    const SE3Pose& getPRE_worldToCam() const {return PRE_worldToCam;}
    void setPRE_camToWorld(const SE3Pose& preCamToWorld) 
    {
        PRE_camToWorld = preCamToWorld;
        PRE_worldToCam = preCamToWorld.inverse();

    }
    void setPRE_worldToCam(const SE3Pose& preWorldToCam) 
    {
        PRE_worldToCam = preWorldToCam;
        PRE_camToWorld = preWorldToCam.inverse();

    }
    EFFrame* efFrame;
    EIGEN_ALWAYS_INLINE Vec6 getPrior() const
    {
        Vec6 p =  Vec6::Zero();        
        if(mKeyFrameId == 0)
        {
            constexpr float setting_initialRotPrior{1e11};
            constexpr float setting_initialTransPrior{1e10};
            p.head<3>() = Vec3::Constant(setting_initialTransPrior);
            p.segment<3>(3) = Vec3::Constant(setting_initialRotPrior);
            if(setting_solverMode & SOLVER_REMOVE_POSEPRIOR) p.head<6>().setZero();
        }
        return p;
    }
    EIGEN_ALWAYS_INLINE Vec6 getPriorZero() {return Vec6::Zero();}
    Vec6 step, step_backup, state_backup;
    void addGoodEdge(EdgePixel* goodEdge) { mEdgesGood.push_back(goodEdge); }
    void setEvalPT_scaled(const SE3Pose &worldToCam_evalPT);
    void setEvalPT(const SE3Pose &worldToCam_evalPT, const Vec6 &state);
    void setState(const Vec6 &state);
    void setStateScaled(const Vec6 &state_scaled);
    void setStateZero(const Vec6 &state_zero);
    size_t idx; //Id in frameHessians
    //if coarseDistanceMap == nullptr -> run without distance map
    void findResidualsToOptimize(DetectedEdgesPtrVec& newEdgePixels, CoarseDistanceMap * const coarseDistanceMap, const FrameData* const refFrameNew, const CameraMatrix& cameraMat, const SystemSettings& config, const float compareDist) const;
    FrameFramePrecalc& getTargetPrecalcById(const size_t id) { return targetPrecalc[id]; }
    const FrameFramePrecalc& getTargetPrecalcById(const size_t id) const { return targetPrecalc[id]; }
    Vec6 nullspaces_scale;
    Mat66 nullspaces_pose;
    bool FIX_WORLD_POSE = false;
    inline Vec6 w2c_leftEps() const {return get_state_scaled().head<6>();}

    //Called during makeKeyframe
    size_t removeGoodEdgesWithoutResiduals();
    void removeOutOfBounds(const WindowedOptimizer& windowedOptimizer, const CameraMatrix& camMat,
                            const std::vector<FrameData*>& fhsToMargPoints,const SystemSettings& config);
    //Called during makeKeyframe end
    void cleanUpPointsAlreadyInWindow();

    cv::Mat returnRgbFullSize() const { return mFrameSet->colorImg; }
    
    char* getFernCode()
    {
        if (mFernCode == nullptr) createFernCode();
        return mFernCode;
    }
    void addConstraint(const CeresConstraint& constraint) { mRelPoseConstraints.push_back(constraint); }
    void removeConstraintsTill(const size_t delTillKfId) 
    { 
        if (mRelPoseConstraints.empty()) return;
        const auto& lastConstraint = mRelPoseConstraints.back();
        //now smaller constraints, thus don't go through all
        if (lastConstraint.i <= delTillKfId && lastConstraint.j <= delTillKfId) return;
        mRelPoseConstraints.erase(std::remove_if(std::begin(mRelPoseConstraints),std::end(mRelPoseConstraints),
                    [id=delTillKfId](const auto& c){return !(c.i <= id && c.j <= id);}),
                    std::end(mRelPoseConstraints));
    }
    const RelPoseConstraints& getConstraints() const {return mRelPoseConstraints;}
    void resetConstraints(){mRelPoseConstraints.clear();}
    cv::Mat reprojDebug;
    void resetFrame();
    bool isInFernDbOrLoop() const { return flagIsInKfDB || flagIsInLoop;}
    bool isInFernDb() const { return flagIsInKfDB;}
    bool isInLoop() const { return flagIsInLoop;}
    void setFlagIsInKfDb(const bool flag) { flagIsInKfDB = flag; };
    void setFlagIsInLoop(const bool flag) { flagIsInLoop = flag; };
    void computeOptimizationStructure(const CameraMatrix& camMat);
    void computeValidEdgePixels(); //Compute a 2D list of valid edge pixels
private:
    bool flagIsInKfDB = false;
    bool flagIsInLoop = false;

    
    void release();
    RelPoseConstraints mRelPoseConstraints; ///< These constraints are the relative transformations to adjacent frames in the window!
   
    bool mOptStructureAlreadyComputed;
    std::array<Vec3f*,PyramidLevels> mOptStructures; ///< Pyramid of DT and the gradients that are primarly used for debugging
    std::vector<cv::Mat> mDtPyramids; ///< Pyramid of down-scaled DT that are primarly used for debugging
    EdgePixelVec mEdgesGood; //successfully traced edges
    EdgePixelVec mEdgesOut;	// contains all OUTLIER points (= discarded.).
    EdgePixelVec mEdgesMarginalized;
    const SystemSettings& mSystemSettings;
    
    void createFernCode()
    {
        if (mFernCode == nullptr)
            mFernCode = new char[FernDBNumFerns];
    }

    char *mFernCode;
    /*PRECALC*/
    // variable info.
    SE3Pose worldToCam_evalPT;
    Vec6 state_zero, state_scaled, state;	// [0-5: worldToCam-leftEps. 6-9: a,b] -> 6-9 not needed!
    FrameFramePrecalcVec targetPrecalc;
    SE3Pose PRE_camToWorld, PRE_worldToCam; //They are always updated!
    /*PRECALC END*/

    //for loop closure!
    // RelPoseConstraints mBaConstraints;
};


struct ImmatureEdgeTemporaryResidual
{
public:
    ResState state_state;
    double state_energy;
    ResState state_NewState;
    double state_NewEnergy;
    FrameData* target;
};


/**
 * @brief The EdgeFrameResidual class
 * Comprises an EdgePixel (edge with optimized depth), host and target frame!
 */
class EdgeFrameResidual
{
    /**
     * @brief EdgeFrameResidual
     * So with char *const a; you have a, which is a const pointer (*) to a char. In other words you can change the char which a is pointing at, but you can't make a point at anything different.
Conversely with const char* b; you have b, which is a pointer (*) to a char which is const. You can make b point at any char you like, but you cannot change the value of that char using *b = ...;.
        * @param host
        * @param target
        * @param edgePixel
        */
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    bool FLAG_DONT_OPT_DEPTH;
    void applyRes(bool copyJacobians);
    EdgeFrameResidual( EdgePixel* const edgePixel_, FrameData* host_, FrameData* target_):
        host(host_),target(target_), edgePixel(edgePixel_)
    {
        efResidual = nullptr;
        instanceCounter++;
        resetOOB();
        J = new RawResidualJacobian();
        assert(((long)J)%16==0);
        isNew = true;
        centerProjectedTo.setZero();
        FLAG_DONT_OPT_DEPTH = false;
    }

    ~EdgeFrameResidual(){
        assert(efResidual == nullptr);
        instanceCounter--;
        delete J;
    }

    EFResidual* efResidual;
    static int instanceCounter;
    RawResidualJacobian* J;

    ResState state_state;
    double state_energy;
    ResState state_NewState;
    double state_NewEnergy;
    double state_NewEnergyWithOutlier;

    //FrameData* and EdgePixel* are just observers! Memory is handled in RevoSystem
    FrameData* host;
    FrameData* target;
    void setState(ResState s) {state_state = s;}
    Vec3f centerProjectedTo;
    bool isNew;
    void resetOOB()
    {
        state_NewEnergy = state_energy = 0;
        state_NewState = ResState::OUTLIER;
        setState(ResState::IN);
    }
    EdgePixel* edgePixel;
    double linearize(const CameraMatrix& cameraMat, const SystemSettings& config);
};

//struct -> because no invariant
struct DetectedEdge
{
    enum class DetectedEdgeStatus
    {
        Valid, //if it was actually used for tracking
        NeverTraced //if it was never used for tracking (e.g. always too far away from an edge)
    };
    
    DetectedEdge(FrameData* host_, const float idepth_, const float x_, const float y_, const float weight_, const Vec3f& color_):
        host(host_),idepth(idepth_),hostX(x_),hostY(y_),weight(weight_),color(color_){
            flagEdgeOOB = false;
            flagAlreadyInWindow = false;
        }

    FrameData* host; ///< ptr to the frame, where the pixel was first discovered, if not kf -> nullptr
    float idepth;
    float hostX,hostY;
    float weight;
    Vec3f color;
    DetectedEdgeStatus edgeStatus = DetectedEdgeStatus::Valid;
    bool flagAlreadyInWindow;// = false; ///< Flag that is true if the edge is already used in the windowed ba;
    bool flagEdgeOOB;// = false;
    auto isValidForOptimizing() const
    {
        return (idepth > 0 && edgeStatus == DetectedEdgeStatus::Valid && !flagAlreadyInWindow);
    }
    double linearizeResidual(const CameraMatrix&  HCalib, const float outlierTHSlack, ImmatureEdgeTemporaryResidual* tmpRes, float &Hdd, float &bd, float idepth,const SystemSettings& config) const;

    EIGEN_ALWAYS_INLINE Vec2f getPixel2D() const { return Vec2f(hostX,hostY); }
    EIGEN_ALWAYS_INLINE Vec3f getPixel2DHom() const { return Vec3f(hostX,hostY,1); }
    EIGEN_ALWAYS_INLINE Vec2f reproject(const Mat33f& KRKi, const Vec3f& Kt) const {return (KRKi*Vec3f(hostX,hostY,1)+Kt*idepth).hnormalized();}
};

class EdgePixel
{
private:
    static int instanceCounter;
    float weight;

    //Note that idepth and idepth_scaled should be identical
    float idepth_scaled;
    float idepth_zero_scaled;
    float idepth_zero;
    float idepth;
    float idepth_backup;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgePixel() = delete;
    bool hasDepthPrior;
    float maxRelBaseline;
    float idepth_hessian;
    float idepth_init;
    size_t numGoodResiduals;
    
    enum EdgeStatus {ACTIVE, INACTIVE, OUTLIER, OOB}; //MARGINALIZED}
    EdgeStatus status;
    inline void setEdgeStatus(EdgeStatus s) {status=s;}
    std::vector<EdgeFrameResidual*> edgeResiduals; //std::vector<PointFrameResidual*> residuals;					// only contains good residuals (not OOB and not OUTLIER). Arbitrary order.
    std::pair<EdgeFrameResidual*, ResState> lastResiduals[2]; 	// contains information about residuals to the last two (!) frames. ([0] = latest, [1] = the one before).
    
    FrameData* host; //convert to weak_ptr at a later stage!
    float hostX,hostY; //x,y coordinates in "host" frame
    Vec3f color;
    
    
    bool projectPixel(const Eigen::Matrix3f& KRK_i, const Eigen::Vector3f& Kt, float &new_x,float &new_y,
                      const float maxWidth, const float maxHeight);
    bool projectPixel(const Eigen::Matrix3f& KRK_i, const Eigen::Vector3f& Kt, Eigen::Vector3f& newPt,
                      const float maxWidth, const float maxHeight);
    explicit EdgePixel(const DetectedEdge &detectedEdge);

    ~EdgePixel()
    {
        assert(efPoint == nullptr);
        release();
    }
    void release();

    EFPoint* efPoint;
    float step;
    float step_backup;
    float nullspaces_scale;

    inline float getIdepth() const {return idepth;}
    inline float getIdepthScaled() const {return idepth_scaled;}
    inline float getIdepthZero() const {return idepth_zero;}
    inline float getIdepthZeroScaled() const {return idepth_zero_scaled;}
    inline float getIdepthBackup() const {return idepth_backup;}
    //Now, we try to fuse
    inline void setIdepthBackup(float iDBackup) {idepth_backup = iDBackup;}
    inline void backupIdepth() {idepth_backup = idepth; }

    inline void setIdepth(float newIDepth) {
        this->idepth = newIDepth;
        this->idepth_scaled = SCALE_IDEPTH * newIDepth;
    }
    inline void setIdepthScaled(float newIDepthScaled) {
        this->idepth = SCALE_IDEPTH_INVERSE * newIDepthScaled;
        this->idepth_scaled = newIDepthScaled;
    }
    inline void setIdepthZero(float newIDepth) {
        idepth_zero = newIDepth;
        idepth_zero_scaled = SCALE_IDEPTH * newIDepth;
        nullspaces_scale = -(newIDepth*1.001 - newIDepth/1.001)*500;
    }

    bool isInlierNew() const;
    bool isOOB(const std::vector<FrameData*>& toMarg) const;
};
}
