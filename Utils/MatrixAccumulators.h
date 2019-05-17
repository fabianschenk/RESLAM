#pragma once
/*
 * computes the outer sum of 10x2 matrices, weighted with a 2x2 matrix:
 * 			H = [x y] * [a b; b c] * [x y]^T
 * (assuming x,y are column-vectors).
 * numerically robust to large sums.
 */
#include <Eigen/Eigen>
#include "../config/Defines.h"

namespace RESLAM
{
template<int i, int j>
class AccumulatorXX
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix<float,i,j> A;
  Eigen::Matrix<float,i,j> A1k;
  Eigen::Matrix<float,i,j> A1m;
  size_t num;

  inline void initialize()
  {
    A.setZero();
    A1k.setZero();
    A1m.setZero();
    num = numIn1 = numIn1k = numIn1m = 0;
  }

  inline void finish()
  {
    shiftUp(true);
    num = numIn1 + numIn1k + numIn1m;
  }


  inline void update(const Eigen::Matrix<float,i,1> &L, const Eigen::Matrix<float,j,1> &R, float w)
  {
      A += w*L*R.transpose();
      numIn1++;
      shiftUp(false);
  }

private:
  float numIn1, numIn1k, numIn1m;

  void shiftUp(bool force)
  {
      if(numIn1 > 1000 || force)
      {
          A1k += A;
          A.setZero();
          numIn1k+=numIn1;
          numIn1=0;
      }
      if(numIn1k > 1000 || force)
      {
          A1m += A1k;
          A1k.setZero();
          numIn1m+=numIn1k;
          numIn1k=0;
      }
  }
};

class AccumulatorApprox
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Mat1111f H; //11 =  6 DOF+ 4camera + 1 column/row for b vector
  size_t num;

  inline void initialize()
  {
    memset(Data,0, sizeof(float)*60);
    memset(Data1k,0, sizeof(float)*60);
    memset(Data1m,0, sizeof(float)*60);

    memset(TopRight_Data,0, sizeof(float)*16);
    memset(TopRight_Data1k,0, sizeof(float)*16);
    memset(TopRight_Data1m,0, sizeof(float)*16);

    BotRight = 0;
    num = numIn1 = numIn1k = numIn1m = 0;
  }



    inline void finish()
    {
        H.setZero();
        shiftUp(true);
        assert(numIn1==0);
        assert(numIn1k==0);

        int idx=0;
        //Since we only computed one the upper triangular part of the symmetric matrix,
        //we have to mirror it
        for(int r=0;r<10;r++)
            for(int c=r;c<10;c++)
            {
                //mirror the matrix!
                H(r,c) = H(c,r) = Data1m[idx];
                idx++;
            }
        //This is for the J_geom and J_photo dependencies (a and b)
        //fills lower left and upper right of the matrix
        idx=0;
        //but we only have 1 instead of three columns
        for(int r=0;r<10;r++)
        {
            int c = 0;
            H(r,c+10) = H(c+10,r) = TopRight_Data1m[idx];
            idx++;
        }

        H(10,10) 			= BotRight; //BotRight_Data1m[5];
        num = numIn1 + numIn1k + numIn1m;
    }

  inline void updateSSE(
          const float* const x,
          const float* const y,
          const float a,
          const float b,
          const float c)
  {
      // E = [a,b;b,c]
      Data[0] += a*x[0]*x[0] + c*y[0]*y[0] +  b*(x[0]*y[0] + y[0]*x[0]); //Data[0] += [x0,y0]E[x0;y0]
      Data[1] += a*x[1]*x[0] + c*y[1]*y[0] +  b*(x[1]*y[0] + y[1]*x[0]);
      Data[2] += a*x[2]*x[0] + c*y[2]*y[0] +  b*(x[2]*y[0] + y[2]*x[0]);
      Data[3] += a*x[3]*x[0] + c*y[3]*y[0] +  b*(x[3]*y[0] + y[3]*x[0]);
      Data[4] += a*x[4]*x[0] + c*y[4]*y[0] +  b*(x[4]*y[0] + y[4]*x[0]);
      Data[5] += a*x[5]*x[0] + c*y[5]*y[0] +  b*(x[5]*y[0] + y[5]*x[0]);
      Data[6] += a*x[6]*x[0] + c*y[6]*y[0] +  b*(x[6]*y[0] + y[6]*x[0]);
      Data[7] += a*x[7]*x[0] + c*y[7]*y[0] +  b*(x[7]*y[0] + y[7]*x[0]);
      Data[8] += a*x[8]*x[0] + c*y[8]*y[0] +  b*(x[8]*y[0] + y[8]*x[0]);
      Data[9] += a*x[9]*x[0] + c*y[9]*y[0] +  b*(x[9]*y[0] + y[9]*x[0]);


      Data[10] += a*x[1]*x[1] + c*y[1]*y[1] +  b*(x[1]*y[1] + y[1]*x[1]);
      Data[11] += a*x[2]*x[1] + c*y[2]*y[1] +  b*(x[2]*y[1] + y[2]*x[1]);
      Data[12] += a*x[3]*x[1] + c*y[3]*y[1] +  b*(x[3]*y[1] + y[3]*x[1]);
      Data[13] += a*x[4]*x[1] + c*y[4]*y[1] +  b*(x[4]*y[1] + y[4]*x[1]);
      Data[14] += a*x[5]*x[1] + c*y[5]*y[1] +  b*(x[5]*y[1] + y[5]*x[1]);
      Data[15] += a*x[6]*x[1] + c*y[6]*y[1] +  b*(x[6]*y[1] + y[6]*x[1]);
      Data[16] += a*x[7]*x[1] + c*y[7]*y[1] +  b*(x[7]*y[1] + y[7]*x[1]);
      Data[17] += a*x[8]*x[1] + c*y[8]*y[1] +  b*(x[8]*y[1] + y[8]*x[1]);
      Data[18] += a*x[9]*x[1] + c*y[9]*y[1] +  b*(x[9]*y[1] + y[9]*x[1]);



      Data[19] += a*x[2]*x[2] + c*y[2]*y[2] +  b*(x[2]*y[2] + y[2]*x[2]);
      Data[20] += a*x[3]*x[2] + c*y[3]*y[2] +  b*(x[3]*y[2] + y[3]*x[2]);
      Data[21] += a*x[4]*x[2] + c*y[4]*y[2] +  b*(x[4]*y[2] + y[4]*x[2]);
      Data[22] += a*x[5]*x[2] + c*y[5]*y[2] +  b*(x[5]*y[2] + y[5]*x[2]);
      Data[23] += a*x[6]*x[2] + c*y[6]*y[2] +  b*(x[6]*y[2] + y[6]*x[2]);
      Data[24] += a*x[7]*x[2] + c*y[7]*y[2] +  b*(x[7]*y[2] + y[7]*x[2]);
      Data[25] += a*x[8]*x[2] + c*y[8]*y[2] +  b*(x[8]*y[2] + y[8]*x[2]);
      Data[26] += a*x[9]*x[2] + c*y[9]*y[2] +  b*(x[9]*y[2] + y[9]*x[2]);



      Data[27] += a*x[3]*x[3] + c*y[3]*y[3] +  b*(x[3]*y[3] + y[3]*x[3]);
      Data[28] += a*x[4]*x[3] + c*y[4]*y[3] +  b*(x[4]*y[3] + y[4]*x[3]);
      Data[29] += a*x[5]*x[3] + c*y[5]*y[3] +  b*(x[5]*y[3] + y[5]*x[3]);
      Data[30] += a*x[6]*x[3] + c*y[6]*y[3] +  b*(x[6]*y[3] + y[6]*x[3]);
      Data[31] += a*x[7]*x[3] + c*y[7]*y[3] +  b*(x[7]*y[3] + y[7]*x[3]);
      Data[32] += a*x[8]*x[3] + c*y[8]*y[3] +  b*(x[8]*y[3] + y[8]*x[3]);
      Data[33] += a*x[9]*x[3] + c*y[9]*y[3] +  b*(x[9]*y[3] + y[9]*x[3]);



      Data[34] += a*x[4]*x[4] + c*y[4]*y[4] +  b*(x[4]*y[4] + y[4]*x[4]);
      Data[35] += a*x[5]*x[4] + c*y[5]*y[4] +  b*(x[5]*y[4] + y[5]*x[4]);
      Data[36] += a*x[6]*x[4] + c*y[6]*y[4] +  b*(x[6]*y[4] + y[6]*x[4]);
      Data[37] += a*x[7]*x[4] + c*y[7]*y[4] +  b*(x[7]*y[4] + y[7]*x[4]);
      Data[38] += a*x[8]*x[4] + c*y[8]*y[4] +  b*(x[8]*y[4] + y[8]*x[4]);
      Data[39] += a*x[9]*x[4] + c*y[9]*y[4] +  b*(x[9]*y[4] + y[9]*x[4]);



      Data[40] += a*x[5]*x[5] + c*y[5]*y[5] +  b*(x[5]*y[5] + y[5]*x[5]);
      Data[41] += a*x[6]*x[5] + c*y[6]*y[5] +  b*(x[6]*y[5] + y[6]*x[5]);
      Data[42] += a*x[7]*x[5] + c*y[7]*y[5] +  b*(x[7]*y[5] + y[7]*x[5]);
      Data[43] += a*x[8]*x[5] + c*y[8]*y[5] +  b*(x[8]*y[5] + y[8]*x[5]);
      Data[44] += a*x[9]*x[5] + c*y[9]*y[5] +  b*(x[9]*y[5] + y[9]*x[5]);


      Data[45] += a*x[6]*x[6] + c*y[6]*y[6] +  b*(x[6]*y[6] + y[6]*x[6]);
      Data[46] += a*x[7]*x[6] + c*y[7]*y[6] +  b*(x[7]*y[6] + y[7]*x[6]);
      Data[47] += a*x[8]*x[6] + c*y[8]*y[6] +  b*(x[8]*y[6] + y[8]*x[6]);
      Data[48] += a*x[9]*x[6] + c*y[9]*y[6] +  b*(x[9]*y[6] + y[9]*x[6]);


      Data[49] += a*x[7]*x[7] + c*y[7]*y[7] +  b*(x[7]*y[7] + y[7]*x[7]);
      Data[50] += a*x[8]*x[7] + c*y[8]*y[7] +  b*(x[8]*y[7] + y[8]*x[7]);
      Data[51] += a*x[9]*x[7] + c*y[9]*y[7] +  b*(x[9]*y[7] + y[9]*x[7]);


      Data[52] += a*x[8]*x[8] + c*y[8]*y[8] +  b*(x[8]*y[8] + y[8]*x[8]);
      Data[53] += a*x[9]*x[8] + c*y[9]*y[8] +  b*(x[9]*y[8] + y[9]*x[8]);

      Data[54] += a*x[9]*x[9] + c*y[9]*y[9] +  b*(x[9]*y[9] + y[9]*x[9]);


      num++;
      numIn1++;
      shiftUp(false);
  }




/*
 * same as other method, just that x/y are composed of two parts, the first 4 elements are in x4/y4, the last 6 in x6/y6.
 */
  inline void update(
          const float* const x4,
          const float* const x6,
          const float* const y4,
          const float* const y6,
          const float a,
          const float b,
          const float c)
  {



      //This is NOT propagation of uncertainty but J'J for Hessian approximation
      //J'J is symmetric, thus we only compute the upper triangular
      //we need a total of 55 elements = 10 + 9 + 8...+2+1
      //dx,dy = image gradients
      //We have: ([dx,dy] * [J_x,J_y])'*([dx,dy]*[J_x,J_y]) = [J_x,J_y]'*[dx,dy]'*[dx,dy]*[J_x,J_y] = [J_x,J_y]'*[a,b;b,c]*[J_x,J_y]
      // E = [a,b;b,c], where a=dx*dx, b = dx*dy, c = dy*dy                      //x4,y4       //x4,y4
      Data[0] += a*x4[0]*x4[0] + c*y4[0]*y4[0] +  b*(x4[0]*y4[0] + y4[0]*x4[0]); //[x0,y0] * E * [x0;y0]
      Data[1] += a*x4[1]*x4[0] + c*y4[1]*y4[0] +  b*(x4[1]*y4[0] + y4[1]*x4[0]); //[x0,y0] * E * [x1;y1]
      Data[2] += a*x4[2]*x4[0] + c*y4[2]*y4[0] +  b*(x4[2]*y4[0] + y4[2]*x4[0]); //[x0,y0] * E * [x2;y2]
      Data[3] += a*x4[3]*x4[0] + c*y4[3]*y4[0] +  b*(x4[3]*y4[0] + y4[3]*x4[0]); //[x0,y0] * E * [x3;y3]
      //We combine each gradient from x6 with x4                                 //x4,y4       //x6,y6
      Data[4] += a*x6[0]*x4[0] + c*y6[0]*y4[0] +  b*(x6[0]*y4[0] + y6[0]*x4[0]); //[x0,y0] * E * [x0;y0]
      Data[5] += a*x6[1]*x4[0] + c*y6[1]*y4[0] +  b*(x6[1]*y4[0] + y6[1]*x4[0]); //[x0,y0] * E * [x1;y1]
      Data[6] += a*x6[2]*x4[0] + c*y6[2]*y4[0] +  b*(x6[2]*y4[0] + y6[2]*x4[0]); //[x0,y0] * E * [x2;y2]
      Data[7] += a*x6[3]*x4[0] + c*y6[3]*y4[0] +  b*(x6[3]*y4[0] + y6[3]*x4[0]); //[x0,y0] * E * [x3;y3]
      Data[8] += a*x6[4]*x4[0] + c*y6[4]*y4[0] +  b*(x6[4]*y4[0] + y6[4]*x4[0]);
      Data[9] += a*x6[5]*x4[0] + c*y6[5]*y4[0] +  b*(x6[5]*y4[0] + y6[5]*x4[0]);




      Data[10] += a*x4[1]*x4[1] + c*y4[1]*y4[1] +  b*(x4[1]*y4[1] + y4[1]*x4[1]); //[x1,y1] * E * [x1;y1]
      Data[11] += a*x4[2]*x4[1] + c*y4[2]*y4[1] +  b*(x4[2]*y4[1] + y4[2]*x4[1]); //[x1,y1] * E * [x2;y2]
      Data[12] += a*x4[3]*x4[1] + c*y4[3]*y4[1] +  b*(x4[3]*y4[1] + y4[3]*x4[1]); //[x1,y1] * E * [x3;y3]
      Data[13] += a*x6[0]*x4[1] + c*y6[0]*y4[1] +  b*(x6[0]*y4[1] + y6[0]*x4[1]);
      Data[14] += a*x6[1]*x4[1] + c*y6[1]*y4[1] +  b*(x6[1]*y4[1] + y6[1]*x4[1]);
      Data[15] += a*x6[2]*x4[1] + c*y6[2]*y4[1] +  b*(x6[2]*y4[1] + y6[2]*x4[1]);
      Data[16] += a*x6[3]*x4[1] + c*y6[3]*y4[1] +  b*(x6[3]*y4[1] + y6[3]*x4[1]);
      Data[17] += a*x6[4]*x4[1] + c*y6[4]*y4[1] +  b*(x6[4]*y4[1] + y6[4]*x4[1]);
      Data[18] += a*x6[5]*x4[1] + c*y6[5]*y4[1] +  b*(x6[5]*y4[1] + y6[5]*x4[1]);



      Data[19] += a*x4[2]*x4[2] + c*y4[2]*y4[2] +  b*(x4[2]*y4[2] + y4[2]*x4[2]);
      Data[20] += a*x4[3]*x4[2] + c*y4[3]*y4[2] +  b*(x4[3]*y4[2] + y4[3]*x4[2]);
      Data[21] += a*x6[0]*x4[2] + c*y6[0]*y4[2] +  b*(x6[0]*y4[2] + y6[0]*x4[2]);
      Data[22] += a*x6[1]*x4[2] + c*y6[1]*y4[2] +  b*(x6[1]*y4[2] + y6[1]*x4[2]);
      Data[23] += a*x6[2]*x4[2] + c*y6[2]*y4[2] +  b*(x6[2]*y4[2] + y6[2]*x4[2]);
      Data[24] += a*x6[3]*x4[2] + c*y6[3]*y4[2] +  b*(x6[3]*y4[2] + y6[3]*x4[2]);
      Data[25] += a*x6[4]*x4[2] + c*y6[4]*y4[2] +  b*(x6[4]*y4[2] + y6[4]*x4[2]);
      Data[26] += a*x6[5]*x4[2] + c*y6[5]*y4[2] +  b*(x6[5]*y4[2] + y6[5]*x4[2]);



      Data[27] += a*x4[3]*x4[3] + c*y4[3]*y4[3] +  b*(x4[3]*y4[3] + y4[3]*x4[3]);
      Data[28] += a*x6[0]*x4[3] + c*y6[0]*y4[3] +  b*(x6[0]*y4[3] + y6[0]*x4[3]);
      Data[29] += a*x6[1]*x4[3] + c*y6[1]*y4[3] +  b*(x6[1]*y4[3] + y6[1]*x4[3]);
      Data[30] += a*x6[2]*x4[3] + c*y6[2]*y4[3] +  b*(x6[2]*y4[3] + y6[2]*x4[3]);
      Data[31] += a*x6[3]*x4[3] + c*y6[3]*y4[3] +  b*(x6[3]*y4[3] + y6[3]*x4[3]);
      Data[32] += a*x6[4]*x4[3] + c*y6[4]*y4[3] +  b*(x6[4]*y4[3] + y6[4]*x4[3]);
      Data[33] += a*x6[5]*x4[3] + c*y6[5]*y4[3] +  b*(x6[5]*y4[3] + y6[5]*x4[3]);



      Data[34] += a*x6[0]*x6[0] + c*y6[0]*y6[0] +  b*(x6[0]*y6[0] + y6[0]*x6[0]);
      Data[35] += a*x6[1]*x6[0] + c*y6[1]*y6[0] +  b*(x6[1]*y6[0] + y6[1]*x6[0]);
      Data[36] += a*x6[2]*x6[0] + c*y6[2]*y6[0] +  b*(x6[2]*y6[0] + y6[2]*x6[0]);
      Data[37] += a*x6[3]*x6[0] + c*y6[3]*y6[0] +  b*(x6[3]*y6[0] + y6[3]*x6[0]);
      Data[38] += a*x6[4]*x6[0] + c*y6[4]*y6[0] +  b*(x6[4]*y6[0] + y6[4]*x6[0]);
      Data[39] += a*x6[5]*x6[0] + c*y6[5]*y6[0] +  b*(x6[5]*y6[0] + y6[5]*x6[0]);



      Data[40] += a*x6[1]*x6[1] + c*y6[1]*y6[1] +  b*(x6[1]*y6[1] + y6[1]*x6[1]);
      Data[41] += a*x6[2]*x6[1] + c*y6[2]*y6[1] +  b*(x6[2]*y6[1] + y6[2]*x6[1]);
      Data[42] += a*x6[3]*x6[1] + c*y6[3]*y6[1] +  b*(x6[3]*y6[1] + y6[3]*x6[1]);
      Data[43] += a*x6[4]*x6[1] + c*y6[4]*y6[1] +  b*(x6[4]*y6[1] + y6[4]*x6[1]);
      Data[44] += a*x6[5]*x6[1] + c*y6[5]*y6[1] +  b*(x6[5]*y6[1] + y6[5]*x6[1]);


      Data[45] += a*x6[2]*x6[2] + c*y6[2]*y6[2] +  b*(x6[2]*y6[2] + y6[2]*x6[2]);
      Data[46] += a*x6[3]*x6[2] + c*y6[3]*y6[2] +  b*(x6[3]*y6[2] + y6[3]*x6[2]);
      Data[47] += a*x6[4]*x6[2] + c*y6[4]*y6[2] +  b*(x6[4]*y6[2] + y6[4]*x6[2]);
      Data[48] += a*x6[5]*x6[2] + c*y6[5]*y6[2] +  b*(x6[5]*y6[2] + y6[5]*x6[2]);


      Data[49] += a*x6[3]*x6[3] + c*y6[3]*y6[3] +  b*(x6[3]*y6[3] + y6[3]*x6[3]);
      Data[50] += a*x6[4]*x6[3] + c*y6[4]*y6[3] +  b*(x6[4]*y6[3] + y6[4]*x6[3]);
      Data[51] += a*x6[5]*x6[3] + c*y6[5]*y6[3] +  b*(x6[5]*y6[3] + y6[5]*x6[3]);


      Data[52] += a*x6[4]*x6[4] + c*y6[4]*y6[4] +  b*(x6[4]*y6[4] + y6[4]*x6[4]);
      Data[53] += a*x6[5]*x6[4] + c*y6[5]*y6[4] +  b*(x6[5]*y6[4] + y6[5]*x6[4]);

      Data[54] += a*x6[5]*x6[5] + c*y6[5]*y6[5] +  b*(x6[5]*y6[5] + y6[5]*x6[5]);


      num++;
      numIn1++;
      shiftUp(false);
  }

  /**
   * @brief updateTopRight
   * @param x4 rJ->Jpdc[0].data()
   * @param x6 rJ->Jpdxi[0].data()
   * @param y4 rJ->Jpdc[1].data()
   * @param y6 rJ->Jpdxi[1].data()
   * @param TR00 rJ->JabJIdx(0,0)
   * @param TR10 rJ->JabJIdx(0,1)
   * @param TR01 rJ->JabJIdx(1,0)
   * @param TR11 rJ->JabJIdx(1,1)
   * @param TR02 JI_r[0]
   * @param TR12 JI_r[1]
   */
  //this is a new version!
  //Since, we only have
//  inline void updateTopRight(
//          const float* const x4,
//          const float* const x6,
//          const float* const y4,
//          const float* const y6,
//          const float TR00, const float TR10,
//          const float TR01, const float TR11,
//          const float TR02, const float TR12 )
//  {
  inline void updateTopRight(
          const float* const x4,
          const float* const x6,
          const float* const y4,
          const float* const y6,
          const float TR02, const float TR12 )
  {
      //If you see TR as [t00 t01 t02; t10 t11 t12]', we can multiply [x4;y4] onto it
      //This is TR'*[x4[0];y4[0]]
//      TopRight_Data[0] += x4[0]*TR00 + y4[0]*TR10;
//      TopRight_Data[1] += x4[0]*TR01 + y4[0]*TR11;
      TopRight_Data[0] += x4[0]*TR02 + y4[0]*TR12;
      //This is TR'*[x4[1];y4[1]]

//      TopRight_Data[3] += x4[1]*TR00 + y4[1]*TR10;
//      TopRight_Data[4] += x4[1]*TR01 + y4[1]*TR11;
      TopRight_Data[1] += x4[1]*TR02 + y4[1]*TR12;

//      TopRight_Data[6] += x4[2]*TR00 + y4[2]*TR10;
//      TopRight_Data[7] += x4[2]*TR01 + y4[2]*TR11;
      TopRight_Data[2] += x4[2]*TR02 + y4[2]*TR12;

//      TopRight_Data[9] += x4[3]*TR00 + y4[3]*TR10;
//      TopRight_Data[10] += x4[3]*TR01 + y4[3]*TR11;
      TopRight_Data[3] += x4[3]*TR02 + y4[3]*TR12;

//      TopRight_Data[12] += x6[0]*TR00 + y6[0]*TR10;
//      TopRight_Data[13] += x6[0]*TR01 + y6[0]*TR11;
      TopRight_Data[4] += x6[0]*TR02 + y6[0]*TR12;

//      TopRight_Data[15] += x6[1]*TR00 + y6[1]*TR10;
//      TopRight_Data[16] += x6[1]*TR01 + y6[1]*TR11;
      TopRight_Data[5] += x6[1]*TR02 + y6[1]*TR12;

//      TopRight_Data[18] += x6[2]*TR00 + y6[2]*TR10;
//      TopRight_Data[19] += x6[2]*TR01 + y6[2]*TR11;
      TopRight_Data[6] += x6[2]*TR02 + y6[2]*TR12;

//      TopRight_Data[21] += x6[3]*TR00 + y6[3]*TR10;
//      TopRight_Data[22] += x6[3]*TR01 + y6[3]*TR11;
      TopRight_Data[7] += x6[3]*TR02 + y6[3]*TR12;

//      TopRight_Data[24] += x6[4]*TR00 + y6[4]*TR10;
//      TopRight_Data[25] += x6[4]*TR01 + y6[4]*TR11;
      TopRight_Data[8] += x6[4]*TR02 + y6[4]*TR12;

//      TopRight_Data[27] += x6[5]*TR00 + y6[5]*TR10;
//      TopRight_Data[28] += x6[5]*TR01 + y6[5]*TR11;
      TopRight_Data[9] += x6[5]*TR02 + y6[5]*TR12;

  }

//Since we don't have a and b, we only have to add the squared residual
inline void updateBotRight(const float r_sq)
{
    BotRight += r_sq;
}



private:
  EIGEN_ALIGN16 float Data[60];
  EIGEN_ALIGN16 float Data1k[60];
  EIGEN_ALIGN16 float Data1m[60];

  //instead of 30 elements, we only have 10 -> to be aligned, we have to allocate 16
  EIGEN_ALIGN16 float TopRight_Data[16];
  EIGEN_ALIGN16 float TopRight_Data1k[16];
  EIGEN_ALIGN16 float TopRight_Data1m[16];
  float BotRight; //this is only the residual squared
  float numIn1, numIn1k, numIn1m;
  void shiftUp(bool force)
  {
      if(numIn1 > 1000 || force)
      {
          for(int i=0;i<60;i+=4)
              _mm_store_ps(Data1k+i, _mm_add_ps(_mm_load_ps(Data+i),_mm_load_ps(Data1k+i)));
          for(int i=0;i<16;i+=4)
              _mm_store_ps(TopRight_Data1k+i, _mm_add_ps(_mm_load_ps(TopRight_Data+i),_mm_load_ps(TopRight_Data1k+i)));

          numIn1k+=numIn1;
          numIn1=0;
          memset(Data,0, sizeof(float)*60);
          memset(TopRight_Data,0, sizeof(float)*16);
          BotRight = 0;
      }

      if(numIn1k > 1000 || force)
      {
          for(int i=0;i<60;i+=4)
              _mm_store_ps(Data1m+i, _mm_add_ps(_mm_load_ps(Data1k+i),_mm_load_ps(Data1m+i)));
          for(int i=0;i<16;i+=4)
              _mm_store_ps(TopRight_Data1m+i, _mm_add_ps(_mm_load_ps(TopRight_Data1k+i),_mm_load_ps(TopRight_Data1m+i)));

          numIn1m+=numIn1k;
          numIn1k=0;
          memset(Data1k,0, sizeof(float)*60);
          memset(TopRight_Data1k,0, sizeof(float)*16);
          BotRight = 0;
      }
  }
};

template<int i>
class AccumulatorX
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix<float,i,1> A;
  Eigen::Matrix<float,i,1> A1k;
  Eigen::Matrix<float,i,1> A1m;
  size_t num;

  inline void initialize()
  {
    A.setZero();
    A1k.setZero();
    A1m.setZero();
    num = numIn1 = numIn1k = numIn1m = 0;
  }

  inline void finish()
  {
    shiftUp(true);
    num = numIn1+numIn1k+numIn1m;
  }


  inline void update(const Eigen::Matrix<float,i,1> &L, float w)
  {
      A += w*L;
      numIn1++;
      shiftUp(false);
  }

  inline void updateNoWeight(const Eigen::Matrix<float,i,1> &L)
  {
      A += L;
      numIn1++;
      shiftUp(false);
  }

private:
  float numIn1, numIn1k, numIn1m;

  void shiftUp(bool force)
  {
      if(numIn1 > 1000 || force)
      {
          A1k += A;
          A.setZero();
          numIn1k+=numIn1;
          numIn1=0;
      }
      if(numIn1k > 1000 || force)
      {
          A1m += A1k;
          A1k.setZero();
          numIn1m+=numIn1k;
          numIn1k=0;
      }
  }
};

class Accumulator11
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  float A;
  size_t num;

  inline void initialize()
  {
    A=0;
    memset(SSEData,0, sizeof(float)*4*1);
    memset(SSEData1k,0, sizeof(float)*4*1);
    memset(SSEData1m,0, sizeof(float)*4*1);
    num = numIn1 = numIn1k = numIn1m = 0;
  }

  inline void finish()
  {
    shiftUp(true);
    A=SSEData1m[0+0] + SSEData1m[0+1] + SSEData1m[0+2] + SSEData1m[0+3];
  }


  inline void updateSingle(
          const float val)
  {
      SSEData[0] += val;
      num++; numIn1++;
      shiftUp(false);
  }

  inline void updateSSE(
          const __m128 val)
  {
      _mm_store_ps(SSEData, _mm_add_ps(_mm_load_ps(SSEData),val));
      num+=4;
      numIn1++;
      shiftUp(false);
  }

  inline void updateSingleNoShift(
          const float val)
  {
      SSEData[0] += val;
      num++; numIn1++;
  }

  inline void updateSSENoShift(
          const __m128 val)
  {
      _mm_store_ps(SSEData, _mm_add_ps(_mm_load_ps(SSEData),val));
      num+=4;
      numIn1++;
  }

private:
  EIGEN_ALIGN16 float SSEData[4*1];
  EIGEN_ALIGN16 float SSEData1k[4*1];
  EIGEN_ALIGN16 float SSEData1m[4*1];
  float numIn1, numIn1k, numIn1m;


  void shiftUp(bool force)
  {
      if(numIn1 > 1000 || force)
      {
          _mm_store_ps(SSEData1k, _mm_add_ps(_mm_load_ps(SSEData),_mm_load_ps(SSEData1k)));
          numIn1k+=numIn1; numIn1=0;
          memset(SSEData,0, sizeof(float)*4*1);
      }

      if(numIn1k > 1000 || force)
      {
          _mm_store_ps(SSEData1m, _mm_add_ps(_mm_load_ps(SSEData1k),_mm_load_ps(SSEData1m)));
          numIn1m+=numIn1k;  numIn1k=0;
          memset(SSEData1k,0, sizeof(float)*4*1);
      }
  }
};
}