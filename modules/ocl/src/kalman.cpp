#include "precomp.hpp"

using namespace std;
using namespace cv;
using namespace cv::ocl;

KalmanFilter::KalmanFilter() {}

KalmanFilter::KalmanFilter(int dynamParams, int measureParams, int controlParams, int type)
{
    init(dynamParams, measureParams, controlParams, type);
}

void KalmanFilter::init(int DP, int MP, int CP, int type)
{
    CV_Assert( DP > 0 && MP > 0 );
    CV_Assert( type == CV_32F || type == CV_64F );
    CP = cv::max(CP, 0);

    statePre.create(DP, 1, type);
    statePre.setTo(Scalar::all(0));
    
    statePost.create(DP, 1, type);
    statePost.setTo(Scalar::all(0));

    transitionMatrix.create(DP, DP, type);
    setIdentity(transitionMatrix, 1);

    processNoiseCov.create(DP, DP, type);
    setIdentity(processNoiseCov, 1);

    measurementNoiseCov.create(MP, MP, type);
    setIdentity(measurementNoiseCov, 1);

    measurementMatrix.create(MP, DP, type);
    measurementMatrix.setTo(Scalar::all(0));

    errorCovPre.create(DP, DP, type);
    errorCovPre.setTo(Scalar::all(0));

    errorCovPost.create(DP, DP, type);
    errorCovPost.setTo(Scalar::all(0));

    gain.create(DP, MP, type);
    gain.setTo(Scalar::all(0));

    if( CP > 0 )
    {
        controlMatrix.create(DP, CP, type);
        controlMatrix.setTo(Scalar::all(0));
    }
    else
        controlMatrix.release();

    temp1.create(DP, DP, type);
    temp2.create(MP, DP, type);
    temp3.create(MP, MP, type);
    temp4.create(MP, DP, type);
    temp5.create(MP, 1, type);
}

CV_EXPORTS const oclMat& KalmanFilter::predict(const oclMat& control)
{
    // update the state: x'(k) = A*x(k) transitionMatrix = A; statePost = x(k)
    gemm(transitionMatrix, statePost, 1, 
        oclMat(), 
        0, statePre);
    //statePre = transitionMatrix*statePost;

    oclMat temp;
    if( control.data ) //control = B; controlMatrix = u(k)
        // x'(k) = x'(k) + B*u(k)
    {
        //statePre += controlMatrix*control;
        gemm(controlMatrix, control, 1, statePre, 1, statePre);
    }

    // update error covariance matrices: temp1 = A*P(k)
    //temp1 = transitionMatrix*errorCovPost; errorCovPost = P(k)
    gemm(transitionMatrix, errorCovPost, 1, 
        oclMat(), 
        0, temp1);
     
    // P'(k) = temp1*At + Q; processNoiseCov = Q
    gemm(temp1, transitionMatrix, 1, processNoiseCov, 1, errorCovPre, GEMM_2_T);

    // handle the case when there will be measurement before the next predict.
    statePre.copyTo(statePost);

    return statePre;
}

CV_EXPORTS const oclMat& KalmanFilter::correct(const oclMat& measurement)
{
    // temp2 = H*P'(k); measurementMatrix = H
    CV_Assert(measurement.empty() == false);
    gemm(measurementMatrix, errorCovPre, 1, 
        oclMat(),
        0, temp2);

    // temp3 = temp2*Ht + R; measurementNoiseCov = R;
    gemm(temp2, measurementMatrix, 1, measurementNoiseCov, 1, temp3, GEMM_2_T);

    // temp4 = inv(temp3)*temp2 = Kt(k)
    Mat temp;

    solve(Mat(temp3), Mat(temp2), temp, DECOMP_SVD);
    temp4.upload(temp);

    // K(k)
    gain = temp4.t();

    // temp5 = z(k) - H*x'(k); measurement = z(k);
    gemm(measurementMatrix, statePre, -1, measurement, 1, temp5);

    // x(k) = x'(k) + K(k)*temp5
    gemm(gain, temp5, 1, statePre, 1, statePost);

    // P(k) = P'(k) - K(k)*temp2
    gemm(gain, temp2, -1, errorCovPre, 1, errorCovPost);

    return statePost;
}