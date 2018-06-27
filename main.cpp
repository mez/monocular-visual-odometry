#include <ctype.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>

/**
 ***************** Algorithm Outline
    1. Capture images: It, It+1,
    2. Undistort the above images. (using kitti so this step is done for you!)
    3. Use FAST algorithm to detect features in It, and track those features to It+1. A new detection is triggered if the number of features drop below a certain threshold.
    4. Use Nister’s 5-point alogirthm with RANSAC to compute the essential matrix.
    5. Estimate R,t from the essential matrix that was computed in the previous step.
    6. Take scale information from some external source (like a speedometer), and concatenate the translation vectors, and rotation matrices.
 *************************/

 using namespace std;
 using cv::Mat;
 using cv::Point2f;
 using cv::KeyPoint;
 using cv::Size;
 using cv::TermCriteria;

#define MAX_FRAME 2000
#define MIN_NUM_FEAT 2000

const static char* dataset_images_location = "/home/mez/dataset/kitti/dataset/sequences/00/image_1";
const static char* dataset_poses_location = "/home/mez/dataset/kitti/odomerty_poses/poses/00.txt";


vector<Point2f> getGreyCamGroundPoses() {
  string line;
  int i = 0;
  ifstream myfile (dataset_poses_location);
  double value = 0;
  vector<Point2f> poses;
  if (myfile.is_open())
  {
    while ( getline (myfile,line)  )
    {
      Point2f pose;
      std::istringstream in(line);
      for (int j=0; j<12; j++)  {
        in >> value;
        if (j==11) pose.y=value;
        if (j==3) pose.x=value;
      }

      poses.push_back(pose);
      i++;
    }
    myfile.close();
  }

  return poses;

}

vector<double> getAbsoluteScales()	{

  vector<double> scales;
  string line;
  int i = 0;
  ifstream myfile (dataset_poses_location);
  double x =0, y=0, z = 0;
  double x_prev, y_prev, z_prev;
  if (myfile.is_open())
  {
    while ( getline (myfile,line)  )
    {
      z_prev = z;
      x_prev = x;
      y_prev = y;
      std::istringstream in(line);
      //cout << line << '\n';
      for (int j=0; j<12; j++)  {
        in >> z ;
        if (j==7) y=z;
        if (j==3)  x=z;
      }

      scales.push_back(sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)));
    }
    myfile.close();
  }

  return scales;
}

 void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status){
   //this function automatically gets rid of points for which tracking fails

   vector<float> err;
   Size winSize(21,21);
   TermCriteria termcrit(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

   calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

   //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
   int indexCorrection = 0;
   for( int i=0; i<status.size(); i++)
   {
     Point2f pt = points2.at(i- indexCorrection);
     if ((status.at(i) == 0) || (pt.x<0) || (pt.y<0))	{
       if((pt.x<0) || (pt.y<0))	{
         status.at(i) = 0;
       }

       points1.erase (points1.begin() + i - indexCorrection);
       points2.erase (points2.begin() + i - indexCorrection);
       indexCorrection++;
     }
   }
 }

 void featureDetection(Mat img, vector<Point2f> & points){
  vector<KeyPoint> keypoints;
  int fast_threshold = 20;
  bool nonmaxSupression = true;

  cv::FAST(img, keypoints, fast_threshold, nonmaxSupression);
  KeyPoint::convert(keypoints, points, vector<int>());
 }


int main(int argc, char** argv) {

  bool renderFeatures = false;
  if (argc > 1) {
    renderFeatures = true;
  }

  Mat img_1, img_2;
  Mat R_f, t_f;


  double scale = 1.00;
  char filename1[200];
  char filename2[200];
  sprintf(filename1, "%s/%06d.png", dataset_images_location, 0);
  sprintf(filename2, "%s/%06d.png", dataset_images_location, 1);


  char text[100];
  int fontFace = cv::FONT_HERSHEY_PLAIN;
  double fontScale = 1;
  int thickness = 1;
  cv::Point textOrg(10, 50);

  //read the first two frames from the dataset
  Mat img_1_c = cv::imread(filename1);
  Mat img_2_c = cv::imread(filename2);

  if ( !img_1_c.data || !img_2_c.data ) {
    std::cout<< " --(!) Error reading images " << std::endl; return -1;
  }

  // we work with grayscale images
  cvtColor(img_1_c, img_1, cv::COLOR_BGR2GRAY);
  cvtColor(img_2_c, img_2, cv::COLOR_BGR2GRAY);

  // feature detection, tracking
  vector<Point2f> points1, points2;        //vectors to store the coordinates of the feature points
  featureDetection(img_1, points1);        //detect features in img_1
  vector<uchar> status;
  featureTracking(img_1,img_2,points1,points2, status); //track those features to img_2

  //TODO: add a fucntion to load these values directly from KITTI's calib files
  // WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic parameters
  /*
   * Projection matrix example after rectification: Right Grey scale Camera (3x4)
   *
   *     7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 -3.861448000000e+02
   *     0.000000000000e+00 7.188560000000e+02 1.852157000000e+02  0.000000000000e+00
   *     0.000000000000e+00 0.000000000000e+00 1.000000000000e+00  0.000000000000e+00
   */
  double focal = 718.8560;
  cv::Point2d pp(607.1928, 185.2157);

  //recovering the pose and the essential matrix
  Mat E, R, t, mask;
  E = findEssentialMat(points2, points1, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
  recoverPose(E, points2, points1, R, t, focal, pp, mask);


  Mat prevImage = img_2;
  Mat currImage;
  vector<Point2f> prevFeatures = points2;
  vector<Point2f> currFeatures;

  char filename[100];

  R_f = R.clone();
  t_f = t.clone();

  clock_t begin = clock();

  cv::namedWindow( "Road facing camera | Top Down Trajectory", cv::WINDOW_AUTOSIZE );// Create a window for display.

  Mat traj = Mat::zeros(600, 1241, CV_8UC3);

  auto groundPoses = getGreyCamGroundPoses();
  auto groundScales = getAbsoluteScales();

  for(int numFrame=2; numFrame < MAX_FRAME; numFrame++) {
    sprintf(filename, "%s/%06d.png", dataset_images_location, numFrame);


    Mat currImage_c = cv::imread(filename);
    cvtColor(currImage_c, currImage, cv::COLOR_BGR2GRAY);
    vector<uchar> status;
    featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);

    E = findEssentialMat(currFeatures, prevFeatures, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
    recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

    Mat prevPts(2, prevFeatures.size(), CV_64F), currPts(2, currFeatures.size(), CV_64F);


    for (int i = 0; i < prevFeatures.size(); i++) {   //this (x,y) combination makes sense as observed from the source code of triangulatePoints on GitHub
      prevPts.at<double>(0, i) = prevFeatures.at(i).x;
      prevPts.at<double>(1, i) = prevFeatures.at(i).y;

      currPts.at<double>(0, i) = currFeatures.at(i).x;
      currPts.at<double>(1, i) = currFeatures.at(i).y;
    }

    //This is cheating because ideally you'd want to figure out a way to get scale, but without this cheat there is a lot of drift.
    scale = groundScales[numFrame];

    //only update the current R and t if it makes sense.
    if ((scale > 0.1) && (t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {

      t_f = t_f + scale * (R_f * t);
      R_f = R * R_f;

    }

    //Make sure we have enough features to track
    if (prevFeatures.size() < MIN_NUM_FEAT) {
      featureDetection(prevImage, prevFeatures);
      featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
    }

    prevImage = currImage.clone();
    prevFeatures = currFeatures;

    int x = int(t_f.at<double>(0)) + 600;
    int y = int(t_f.at<double>(2)) + 100;

    circle(traj, cv::Point(x, y), 1, CV_RGB(255, 0, 0), 2);
    circle(traj, cv::Point(groundPoses[numFrame].x+600, groundPoses[numFrame].y+100), 1, CV_RGB(0, 255, 0), 2);

    rectangle(traj, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);
    sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1),t_f.at<double>(2));
    putText(traj, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);

    if (renderFeatures){
      //Draw features as markers for fun
      for(auto point: currFeatures)
        cv::drawMarker(currImage_c, cv::Point(point.x,point.y), CV_RGB(0,255,0), cv::MARKER_TILTED_CROSS, 2, 1, cv::LINE_AA);
    }

    Mat concated;
    cv::vconcat(currImage_c, traj, concated);

    imshow("Road facing camera | Top Down Trajectory", concated);

    cv::waitKey(1);
  }

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout << "Total time taken: " << elapsed_secs << "s" << endl;

  //let the user press a key to exit.
  cv::waitKey(0);

  return 0;
}