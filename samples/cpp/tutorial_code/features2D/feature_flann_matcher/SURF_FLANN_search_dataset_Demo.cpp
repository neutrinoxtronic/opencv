// SURF_FLANN_search_dataset_Demo.cpp
// Naive program to search a query picture in a dataset illustrating usage of FLANN

#include <iostream>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/core/utils/filesystem.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/flann.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

#define _ORB_

const char* keys =
    "{ help h |  | Print help message. }"
    "{ dataset | | Path to the images folder used as dataset. }"
    "{ image |   | Path to the image to search for in the dataset. }";

struct img_info {
    int img_index;
    unsigned int nbr_of_matches;
};


int main( int argc, char* argv[] )
{
    CommandLineParser parser( argc, argv, keys );
    Mat img = imread( samples::findFile( parser.get<String>("image") ), IMREAD_GRAYSCALE );
    if ( img.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        parser.printMessage();
        return -1;
    }

    const cv::String db_path = parser.get<String>("dataset");
    if (!utils::fs::isDirectory(db_path))
    {
        cout << "The dataset folder doesn't exist!\n" << endl;
        return -1;
    }

    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors...
#ifdef _SURF_
    int minHessian = 400;
    Ptr<Feature2D> detector = SURF::create( minHessian );
#elif defined(_ORB_)
    Ptr<Feature2D> detector = ORB::create();
#else
    cout << "Missing or unknown defined descriptor" << endl;
    return -1;
#endif

    // ...for the query image...
    std::vector<KeyPoint> img_keypoints;
    Mat img_descriptors;
    detector->detectAndCompute( img, noArray(), img_keypoints, img_descriptors );

    // ...and in the folder containing the images of the dataset
    std::vector<KeyPoint> db_keypoints;
    Mat db_descriptors;
    std::vector<unsigned int> db_images_indice_range; //store the range of indices per image
    std::vector<int> db_indice_2_image_lut;  //match descriptor indice to its image

    db_images_indice_range.push_back(0);
    std::vector<cv::String> files;
    utils::fs::glob(db_path, cv::String(), files);
    for (std::vector<cv::String>::iterator itr = files.begin(); itr != files.end(); ++itr)
    {
        Mat tmp_img = imread( *itr, IMREAD_GRAYSCALE );
        if (!tmp_img.empty())
        {
            std::vector<KeyPoint> kpts;
            Mat descriptors;
            detector->detectAndCompute( tmp_img, noArray(), kpts, descriptors );

            db_keypoints.insert( db_keypoints.end(), kpts.begin(), kpts.end() );
            db_descriptors.push_back( descriptors );
            db_images_indice_range.push_back( db_images_indice_range.back() + kpts.size() );
        }
    }

    //-- Set the LUT
    db_indice_2_image_lut.resize( db_images_indice_range.back() );
    const int nbr_of_imgs = db_images_indice_range.size()-1;
    for (int i = 0; i < nbr_of_imgs; ++i)
    {
        const unsigned int first_indice = db_images_indice_range[i];
        const unsigned int last_indice = db_images_indice_range[i+1];
        std::fill( db_indice_2_image_lut.begin() + first_indice,
                   db_indice_2_image_lut.begin() + last_indice,
                   i );
    }

    //-- Step 2: build the structure storing the descriptors
#if defined(_SIFT_) || defined(_SURF_)
    flann::GenericIndex<cvflann::L2<float> > index(db_descriptors,
                                                   cvflann::KDTreeIndexParams(4));

#elif defined(_ORB_) || defined(_BRISK_) || defined(_FREAK_) || defined(_AKAZE_)
    /* in case of 'anyimpl::bad_any_cast', requires the get_param fix in LshIndex ctor */
    flann::GenericIndex<cvflann::Hamming<unsigned char> > index(db_descriptors,
                                                                cvflann::LshIndexParams());
#else
    cout<< "Descriptor not listed. Set the proper FLANN distance for this descriptor" <<endl;
    return -1;
#endif

    //-- Step 3: retrieve the descriptors in the dataset matching the ones of the query image
    // /!\ knnSearch doesn't follow OpenCV standards by not initialising empty Mat properties
    const int knn = 2;
    Mat indices(img_descriptors.rows, knn, CV_32S);
#if defined(_SIFT_) || defined(_SURF_)
    Mat dists(img_descriptors.rows, knn, CV_32F);
#elif defined(_ORB_) || defined(_BRISK_) || defined(_FREAK_) || defined(_AKAZE_)
    Mat dists(img_descriptors.rows, knn, CV_32S);
#endif
    index.knnSearch( img_descriptors, indices, dists, knn, cvflann::SearchParams(32) );

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches; //contains
    std::vector<unsigned int> matches_per_img_histogram( nbr_of_imgs, 0 );
    for (int i = 0; i < dists.rows; ++i)
    {
        if (dists.at<float>(i,0) < ratio_thresh * dists.at<float>(i,1))
        {
            const int indice_in_db = indices.at<int>(i,0);
            good_matches.push_back( DMatch{i,
                                           indice_in_db,
                                           db_indice_2_image_lut[indice_in_db],
                                           dists.at<float>(i,0)} );
            matches_per_img_histogram[ db_indice_2_image_lut[indice_in_db] ]++;
        }
    }

    //-- Step 4: find the dataset image with the highest proportion of matches
    std::multimap<float, img_info> images_infos;
    for (int i = 0; i < nbr_of_imgs; ++i)
    {
        const unsigned int nbr_of_matches = matches_per_img_histogram[i];
        if (nbr_of_matches < 4) //we need at leat 4 points for a homography
            continue;

        const unsigned int nbr_of_kpts = db_images_indice_range[i+1] - db_images_indice_range[i];
        const float inverse_proportion_of_retrieved_kpts =
                static_cast<float>(nbr_of_kpts) / static_cast<float>(nbr_of_matches);

        images_infos.insert( std::pair<float,img_info>(inverse_proportion_of_retrieved_kpts,
                                                       img_info{i, nbr_of_matches}) );
    }

    if (images_infos.begin() == images_infos.end())
    {
        cout<<"No good match could be found."<<endl;
        return 0;
    }

    //-- if there are several images with a similar proportion of matches,
    // select the one with the highest number of matches weighted by the
    // squared ratio of proportions
    const float best_matches_proportion = images_infos.begin()->first;
    float new_matches_proportion = best_matches_proportion;
    img_info best_img = images_infos.begin()->second;

    std::multimap<float, img_info>::iterator it = images_infos.begin();
    ++it;
    while ((it->first < 1.1*best_matches_proportion) && (it!=images_infos.end()))
    {
        const float ratio = new_matches_proportion / it->first;
        if( it->second.nbr_of_matches * (ratio * ratio) > best_img.nbr_of_matches)
        {
            new_matches_proportion = it->first;
            best_img = it->second;
        }
        ++it;
    }

    //-- Step 5: filter goodmatches that belong to the best image match of the dataset
    std::vector<DMatch> filtered_good_matches;
    for (std::vector<DMatch>::iterator itr(good_matches.begin()); itr != good_matches.end(); ++itr)
    {
        if (itr->imgIdx == best_img.img_index)
            filtered_good_matches.push_back(*itr);
    }

    //-- Retrieve the best image match from the dataset
    Mat db_img = imread( files[best_img.img_index], IMREAD_GRAYSCALE );

    //-- Draw matches
    Mat img_matches;
    drawMatches( img, img_keypoints, db_img, db_keypoints, filtered_good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //-- Show detected matches
    imshow("Good Matches", img_matches );

    while(1)
    {
        int k = waitKey();
        if (k == 27) // Esc key
            break;
    }
    return 0;
}
#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif
