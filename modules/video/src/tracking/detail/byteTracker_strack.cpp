// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// #include "bytetracker_strack.hpp"
//#include "opencv2/video/detail/bytetracker_strack.hpp"
#include "opencv2/video/tracking.hpp"
//#include "opencv2/core.hpp"

namespace cv {
//namespace detail {
//namespace tracking {

//using namespace cv::detail::tracking;

Strack::Strack()
{
    trackId_ = 0;
    classId_ = 0;
    state_ = TrackState::NEW;
    trackletLen_ = 0;
    startFrame_ = 0;
    kalmanFilter_ = cv::KalmanFilter(8,4);
}

Strack::~Strack()
{
    //nothing
}

Strack::Strack(cv::Rect tlwh,  int classId, float score) : tlwh_(tlwh), classId_(classId), score_(score)
{
    trackId_ = 0;
    state_ = TrackState::NEW;
    trackletLen_ = 0;
    startFrame_ = 0;
    kalmanFilter_ = cv::KalmanFilter(8,4);
}


int Strack::getId() const
{
  return trackId_;
}

cv::Rect Strack::getTlwh() const
{
  return tlwh_;
}

void Strack::setTlwh(cv::Rect tlwh)
{
    tlwh_ = tlwh;
}

TrackState Strack::getState() const
{
    return state_;
}

void Strack::setState(TrackState state)
{
    state_ = state;
}

int Strack::getClass()
{
    return classId_;
}

void Strack::activate(int frame, int id)
{
    startFrame_ = frame;
    trackletLen_ = 0;
    state_ = TrackState::TRACKED;
    trackId_ = id;

    kalmanFilter_.measurementMatrix = cv::Mat::eye(4, 8, CV_32F); //H mat

    cv::Mat_<float> transitionMatrix(8,8); //make it a createTransitionMatrix() or maybe a big wrapper for everything.
    for (int row = 0; row < 8; ++row)
    {
        for (int col = 0; col < 8; ++col)
        {
            if (row == col || col == (row + 4))
                transitionMatrix(row,col) = 1.0;
            else
                transitionMatrix(row,col) = 0.0;
        }
    }
    kalmanFilter_.transitionMatrix = transitionMatrix; // F mat

    //Q matrix
    setIdentity(kalmanFilter_.processNoiseCov, cv::Scalar::all(1e-2));
    for (int i=0; i < 4; i++){
        kalmanFilter_.processNoiseCov.at<float>(i,i) *= 10000;
    }
    kalmanFilter_.processNoiseCov.at<float>(7,7) = static_cast<float>(1e-4);
    //vx  vy = 1
    kalmanFilter_.processNoiseCov.at<float>(4,4) = static_cast<float>(1e-4);
    kalmanFilter_.processNoiseCov.at<float>(5,5) = static_cast<float>(1e-4);

    float cx = static_cast<float>(tlwh_.x + tlwh_.width);
    float cy = static_cast<float>(tlwh_.y + tlwh_.height);
    float w = static_cast<float>(tlwh_.width);
    float h = static_cast<float>(tlwh_.height);
    kalmanFilter_.statePre = (cv::Mat_<float>(8,1,CV_32F) << cx, cy, w, h, 0, 0, 0, 0);
    kalmanFilter_.statePost = (cv::Mat_<float>(8,1,CV_32F) << cx, cy, w, h, 0, 0, 0, 0);

}

void Strack::update(Strack& track)
{
    trackletLen_++;

    float cx = static_cast<float>(track.tlwh_.x + track.tlwh_.width);
    float cy = static_cast<float>(track.tlwh_.y + track.tlwh_.height);
    float w = static_cast<float>(track.tlwh_.width);
    float h = static_cast<float>(track.tlwh_.height);

    cv::Mat measurement = (cv::Mat_<float>(4,1) << cx, cy, w, h);
    kalmanFilter_.correct(measurement);
    score_ = track.score_;

}

void Strack::reactivate(Strack& track, int frame)
{
    update(track);
    startFrame_ = frame;
    trackletLen_ = 0;
    state_ = TrackState::TRACKED;
}

void Strack::incrementTrackletLen()
{
    trackletLen_++;
}

int Strack::getTrackletLen() const
{
    return trackletLen_;
}
void Strack::setTrackletLen(int val)
{
    trackletLen_= val;
}

cv::Rect Strack::predict()
{
    cv::Mat predictionMat = kalmanFilter_.predict();
    cv::Rect prediction(
        static_cast<int>(predictionMat.at<float>(0)),
        static_cast<int>(predictionMat.at<float>(1)),
        static_cast<int>(predictionMat.at<float>(2)),
        static_cast<int>(predictionMat.at<float>(3))
    );
    return prediction;
}

float Strack::getScore() const
{
    return score_;
}

}