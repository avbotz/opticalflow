#include <vector>
#include <iostream>
//#include <time.h>
//#include <random>

#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

#define PI 3.1416

typedef std::pair<float, float> PolVec;
typedef std::vector<PolVec> PolVecs;

const bool debug = true;

const float anglemargin = 0.3;
const float sizemargin = 15;

const int cropx = 200;
const int cropy = 200;

const float fovx = (60/360) * 2 * PI;
const float fovy = fovx;

const float depth = 5;

PolVecs convertPolar(std::vector<cv::Point2f> &first, std::vector<cv::Point2f> &second)
{
    PolVecs vals;
    for(int i = 0; i < first.size(); i++)
    {
        float diffx = second[i].x - first[i].x;
        //Change +y to be up the image instead of down
        float diffy = first[i].y - second[i].y;
        float len = sqrt(pow(diffx, 2) + pow(diffy, 2));
        float angle = atan2(diffy, diffx);
        //For debugging
        if(debug) std::cout << diffx << "\t" << diffy << "\t" << angle << "  \t" << len << "\n";
        vals.push_back(PolVec(angle, len));
    }
    return vals;
}

bool inside(PolVec a, PolVec b)
{
    if(std::max(a.first - b.first, b.first - a.first) < anglemargin ||
        std::max(a.first + 2*PI - b.first, b.first - a.first - 2*PI) < anglemargin ||
        std::max(a.first - b.first - 2*PI, b.first + 2*PI - a.first) < anglemargin)
    {
        if(std::max(a.second - b.second, b.second - a.second) < sizemargin)
        {
            return true;
        }
    }
    return false;
}

PolVec avg(PolVecs &vecs, PolVec &center)
{
    float suma = 0; float sumb = 0; int num = 0;
    for(int i = 0; i < vecs.size(); i++)
    {
        if(inside(center, vecs[i]))
        {
            suma += vecs[i].first;
            sumb += vecs[i].second;
            num++;
        }
    }
    return PolVec(suma/num, sumb/num);
}

int numAround(PolVecs &vecs, PolVec &center)
{
    int num = 0;
    for(int i = 0; i < vecs.size(); i++)
    {
        if(inside(center, vecs[i]))
        {
            num++;
        }
    }
    return num;
}

PolVec bestFit(PolVecs vecs)
{
    int best;
    int bestnum = 0;
    //srand(time(NULL));
    for(int i = 0; i < vecs.size(); i++)
    {
        //TODO: Decide whether to iterate through all points or choose random ones
        //int r = rand() % vecs.size();
        int rn = numAround(vecs, vecs[i]);
        //For debugging
        if(debug) std::cout << rn << "\n";
        if(rn > bestnum){bestnum = rn; best = i;}
    }
    return avg(vecs, vecs[best]);
}

//TODO: Put everything in a loop and take images through stdin
int main(int argc, char* argv[])
{
    cv::Mat img1, img2, img1c, img2c, img1b, img2b;
    std::vector<cv::Point2f> first, second;
    std::vector<uchar> status;
    std::vector<float> error;
    int numfeatures;
    float rotation;
    int centerx, centery;

    if(argc < 4) {std::cout << "not enough args\n"; return 0;}

    if(argc < 5) {rotation = 0;}
    else {rotation = atof(argv[4]);}

    //Read input: image 1, image 2, number of features
    img1 = cv::imread(argv[1]);
    img2 = cv::imread(argv[2]);
    numfeatures = atoi(argv[3]);
    centerx = img1.cols/2;
    centery = img1.rows/2;

    //Rotate second img
    if((int) rotation % 360)
    {
        cv::Mat rot = cv::getRotationMatrix2D(cv::Point(centerx, centery), rotation, 1);
        cv::Mat imgrot;
        cv::warpAffine(img2, imgrot, rot, img2.size());
        img2 = imgrot;
    }

    //Crop to get rid of warped area
    //TODO: Run through image undistort instead
    img1c = img1(cv::Rect(centerx - cropx/2, centery - cropy/2,
                            centerx + cropx/2, centery + cropy/2));
    img2c = img2(cv::Rect(centerx - cropx/2, centery - cropy/2,
                            centerx + cropx/2, centery + cropy/2));


    //Convert to grayscale
    cv::cvtColor(img1c, img1b, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2c, img2b, cv::COLOR_BGR2GRAY);

    //Find features to track
    cv::goodFeaturesToTrack(img1b, first, numfeatures, 0.3, 7);
    cv::goodFeaturesToTrack(img2b, second, numfeatures, 0.3, 7);

    //Calculate optical flow
    cv::calcOpticalFlowPyrLK(img1b, img2b, first, second, status, error);

    //Get rid of points that are not being tracked
    //TODO: I don't think this is necessary, so taking out for now.
    /*
    for(int i = numfeatures - 1; i >= 0; i--)
    {
        if(status[i] != 1)
        {
            std::cout << "erasing...\n";
            first.erase(first.begin() + i);
            second.erase(second.begin() + i);
        }
    }
    */

    //TODO: Calculate change in distance between points to calculate change in depth.
    //      This might be very hard.

    //Find best fit vector with angle and magnitude
    PolVec result = bestFit(convertPolar(first, second));
    
    if(debug) std::cout << result.first << "\t\t" << result.second << "\n";

    //For now assume constant depth
    //TODO: Get distance traveled in relative X and Y direction using actual depth.
    float distx = sin(result.second * cos(result.first) / (float) cropx * fovx) * depth;
    float disty = sin(result.second * sin(result.first) / (float) cropy * fovy) * depth;

    std::cout << distx << "\t\t" << disty << "\n";

    //For debugging
    if(debug)
    {
        for(int i = 0; i < numfeatures; i++)
        {
            if(status[i] == 1)
            {
                cv::line(img2b, first[i], second[i], CV_RGB(255, 0, 0));
            }
        }

        cv::imshow("IMGFlow", img2b);
        cv::waitKey(0);
    }

    return 0;
}
