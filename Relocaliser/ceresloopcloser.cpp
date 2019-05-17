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

#include "ceresloopcloser.h"
#include <fstream>
#include "../Utils/Logging.h"
#include "../System/SystemSettings.h"

namespace RESLAM
{
// Constructs the nonlinear least squares optimization problem from the pose
// graph constraints.
void 
CeresLoopCloser::BuildOptimizationProblem(CeresPoseVector& ceresVectorPoses, const RelPoseConstraints& ceresConstraints, ceres::Problem* problem)
{
    CHECK(problem != NULL);
    if (ceresVectorPoses.empty()) {
        LOG(INFO) << "No poses, no problem to optimize.";
        return;
    }
    if (ceresConstraints.empty()) {
        LOG(INFO) << "No constraints, no problem to optimize.";
        return;
    }
    
    //Loss Function can be changed!
    // ceres::LossFunction* loss_function = new ceres::CauchyLoss(1.0);
    // ceres::LossFunction* loss_function = new ceres::HuberLoss(0.2);
    ceres::LossFunction* loss_function = nullptr;

    ceres::LocalParameterization* quaternion_local_parameterization = new ceres::EigenQuaternionParameterization;
    //The cholesky decomposition of the identity is identity!
    //const Eigen::Matrix<double, 6, 6> sqrt_information = constraint.information.llt().matrixL();
    I3D_LOG(i3d::info) << "ceresVectorPoses: " << ceresVectorPoses.size();
    std::vector<int> added(ceresVectorPoses.size());
    std::fill(added.begin(),added.end(),0);
    LOG_THRESHOLD(i3d::detail);
    for (const auto& c : ceresConstraints)
    {
        const Mat66 sqrt_information = Mat66::Identity();
        ceres::CostFunction* cost_function = PoseGraph3dErrorTerm::Create(c.T_j_i, sqrt_information);

        I3D_LOG(i3d::detail) << "c.T_j_i" << c.T_j_i.unit_quaternion().coeffs().transpose()
                             << " Eigen quat: " << Eigen::Quaterniond(c.T_j_i.rotationMatrix()).coeffs().transpose();
        CeresPose& T_W_i = ceresVectorPoses[c.j];
        CeresPose& T_W_j = ceresVectorPoses[c.i];
        added[c.j]++;
        added[c.i]++;
    
        I3D_LOG(i3d::detail) << "Adding residual block: " << c.i << " to " << c.j;
        I3D_LOG(i3d::detail) << " with: " << T_W_i.t.data() << " and "
                << T_W_j.t.data()
                << "T_w_j: " << T_W_j.returnPose().matrix3x4() << " T_W_i: " << T_W_i.returnPose().matrix3x4()
                << c.T_j_i.matrix3x4();                
        problem->AddResidualBlock(cost_function, loss_function,
                                  T_W_i.t.data(),T_W_i.q.coeffs().data(),
                                  T_W_j.t.data(),T_W_j.q.coeffs().data());
        problem->SetParameterization(T_W_i.q.coeffs().data(), quaternion_local_parameterization);
        problem->SetParameterization(T_W_j.q.coeffs().data(), quaternion_local_parameterization);
    }
    I3D_LOG(i3d::info) << "Setting block 0 constant";
    LOG_THRESHOLD(i3d::info);

    //Set first two frames constant to avoid gauge and scale freedom
    if (mSystemSettings.LoopClosureFixWindowPoses && mSystemSettings.EnableLocalMapping)
    {
        size_t nFixedPoses{0};
        for (auto& p : ceresVectorPoses)
        {
            const auto idx = &p - &ceresVectorPoses[0];
            I3D_LOG(i3d::info) << "p: " << idx;
            if (added[idx] > 0 && p.flagFixPose)
            {
                I3D_LOG(i3d::info) << "Fixing: " << idx << " flagFixPose: " << p.flagFixPose;
                problem->SetParameterBlockConstant(p.t.data());
                problem->SetParameterBlockConstant(p.q.coeffs().data());
                nFixedPoses++;
            }
        }
        I3D_LOG(i3d::info) << "nFixedPoses: " << nFixedPoses;
    }
    else
    {
        //The first frame is always constant
        CeresPose& T_W_0 = ceresVectorPoses[0];
        problem->SetParameterBlockConstant(T_W_0.t.data());
        problem->SetParameterBlockConstant(T_W_0.q.coeffs().data());

        //TODO: keep a list of "real" keyframes???
        for (size_t idx = 1; idx < ceresVectorPoses.size();++idx)
        {
            if (added[idx] > 0)
            {
                I3D_LOG(i3d::info) << "Setting block " << idx <<" constant";
                CeresPose& T_W_1 = ceresVectorPoses[idx];
                problem->SetParameterBlockConstant(T_W_1.t.data());
                problem->SetParameterBlockConstant(T_W_1.q.coeffs().data());
                I3D_LOG(i3d::info) << "Setting block " << idx <<" set constant";
                break;
            }
        }
    }
}

// Returns true if the solve was successful.
bool 
CeresLoopCloser::SolveOptimizationProblem(ceres::Problem* problem)
{
    CHECK(problem != NULL);

    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = 1;
    ceres::Solver::Summary summary;
    ceres::Solve(options, problem, &summary);
    std::cout << summary.FullReport() << '\n';
    return summary.IsSolutionUsable();
}

// Output the poses to the file with format: id x y z q_x q_y q_z q_w.
bool 
CeresLoopCloser::OutputPoses(const std::string& filename, const MapOfPoses& poses)
{
    std::fstream outfile;
    outfile.open(filename.c_str(), std::istream::out);
    if (!outfile) {
        LOG(ERROR) << "Error opening the file: " << filename;
        return false;
    }
    for (std::map<int, Pose3d, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Pose3d> > >::const_iterator poses_iter = poses.begin(); poses_iter != poses.end(); ++poses_iter) {
        const std::map<int, Pose3d, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Pose3d> > >::value_type& pair = *poses_iter;
        outfile << pair.first << " " << pair.second.p.transpose() << " "
                << pair.second.q.x() << " " << pair.second.q.y() << " "
                << pair.second.q.z() << " " << pair.second.q.w() << '\n';
    }
    return true;
}
}