
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;


void templateMatchingOpencvPreparedFunction(Mat img_temp, Mat img_orig) {
	cout << "templateMatchingOpencvPreparedFunction" << endl;

	Mat res;
	Mat img_gray;
	double minVal; double maxVal; Point minLoc; Point maxLoc;

	/* convert the image gray from rgb*/
	cvtColor(img_orig, img_gray, COLOR_BGR2GRAY);
	cvtColor(img_temp, img_temp, COLOR_BGR2GRAY);
	
	/* take the information of width and height of image */
	int w = img_temp.cols;
	int h = img_temp.rows;

	/* Look for a any matching are on image */
	matchTemplate(img_gray, img_temp, res, TM_CCORR);
	minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	/* take the coordinates */
	int x = maxLoc.x;
	int y = maxLoc.y;
	int	top_left[2] = { x, y };
	int top_right[2] = { x + w , y };
	int bottom_left[2] = { x, y + h };
	int bottom_right[2] = { x + w , y + h };
	/* print out coordinates*/
	cout << "top left coordinates: ";
	for (int i = 0; i < 2; i++) {
		cout << top_left[i] << " ";
	}
	cout << "\n" << "top right coordinates: ";
	for (int i = 0; i < 2; i++) {
		cout << top_right[i] << " ";
	}
	cout << "\n" << "bottom left coordinates: ";
	for (int i = 0; i < 2; i++) {
		cout << bottom_left[i] << " ";
	}
	cout << "\n" << "bottom right coordinates: ";
	for (int i = 0; i < 2; i++) {
		cout << bottom_right[i] << " ";
	}
	cout << "-----------------------------------------------\n" << endl;
}

void templateMatchingEqual(Mat img_temp, Mat img_orig) {
	cout << "templateMatchingEqual" << endl;
	Mat img_comp;

	/* convert the image gray from rgb*/
	cvtColor(img_orig, img_comp, COLOR_BGR2GRAY);
	cvtColor(img_temp, img_temp, COLOR_BGR2GRAY);

	/* take the information of width and height of images */
	int width_temp = img_temp.cols;
	int height_temp = img_temp.rows;
	int width_comp = img_comp.cols;
	int height_comp = img_comp.rows;

	/* Look for a any matching are on image */
	cout << "Calculating..." << endl;
	Mat result, image; 
	int x,y;
	for (int w = 0; w < width_comp - width_temp; w++) {
		for (int h = 0; h < height_comp - height_temp; h++) {
			image = img_comp(Rect(w, h, width_temp, height_temp));
			compare(image, img_temp, result, CMP_EQ);
			int countNonZero_value = countNonZero(result);
			if (countNonZero_value == width_temp * height_temp) {
				x = w;
				y = h;
			}
		}
	}
	/* if any matching, take the coordinates */
	if (x != NULL) {
		int	top_left[2] = { x, y };
		int top_right[2] = { x + width_temp , y };
		int bottom_left[2] = { x, y + height_temp };
		int bottom_right[2] = { x + width_temp , y + height_temp };

		cout << "top left coordinates: ";
		for (int i = 0; i < 2; i++) {
			cout << top_left[i] << " ";
		}
		cout << "\n" << "top right coordinates: ";
		for (int i = 0; i < 2; i++) {
			cout << top_right[i] << " ";
		}
		cout << "\n" << "bottom left coordinates: ";
		for (int i = 0; i < 2; i++) {
			cout << bottom_left[i] << " ";
		}
		cout << "\n" << "bottom right coordinates: ";
		for (int i = 0; i < 2; i++) {
			cout << bottom_right[i] << " ";
		}
	}
	cout << "-----------------------------------------------\n" << endl;
}

void templateMatchingMean(Mat img_temp, Mat img_orig) {
	cout << "templateMatchingMean" << endl;
	Mat img_comp;
	/* convert the image gray from rgb*/
	cvtColor(img_orig, img_comp, COLOR_BGR2GRAY);
	cvtColor(img_temp, img_temp, COLOR_BGR2GRAY);

	/* take the information of width and height of images */
	int width_temp = img_temp.cols;
	int height_temp = img_temp.rows;
	int width_comp = img_comp.cols;
	int height_comp = img_comp.rows;

	/* Look for a any matching are on image */
	cout << "Calculating..." << endl;
	Mat difference, image;
	int x, y;
	for (int w = 0; w < width_comp - width_temp; w++) {
		for (int h = 0; h < height_comp - height_temp; h++) {
			image = img_comp(Rect(w, h, width_temp, height_temp));
			/*compare(image, img_temp, result, CMP_EQ);*/
			absdiff(image, img_temp, difference);
			Scalar tempVal = mean(difference);
			double value = tempVal[0];
			if (value == 0) {
				x = w;
				y = h;
			}
		}
	}
	/* if any matching, take the coordinates */
	if (x != NULL) {
		int	top_left[2] = { x, y };
		int top_right[2] = { x + width_temp , y };
		int bottom_left[2] = { x, y + height_temp };
		int bottom_right[2] = { x + width_temp , y + height_temp };

		cout << "top left coordinates: ";
		for (int i = 0; i < 2; i++) {
			cout << top_left[i] << " ";
		}
		cout << "\n" << "top right coordinates: ";
		for (int i = 0; i < 2; i++) {
			cout << top_right[i] << " ";
		}
		cout << "\n" << "bottom left coordinates: ";
		for (int i = 0; i < 2; i++) {
			cout << bottom_left[i] << " ";
		}
		cout << "\n" << "bottom right coordinates: ";
		for (int i = 0; i < 2; i++) {
			cout << bottom_right[i] << " ";
		}
	}
	cout << "-----------------------------------------------\n" << endl;
}

void featureMatchingSirf(Mat img_temp, Mat img_orig){
	cout << "featureMatchingSirf" << endl;
	Mat img_comp;

	cvtColor(img_orig, img_comp, COLOR_BGR2GRAY);
	cvtColor(img_temp, img_temp, COLOR_BGR2GRAY);

	vector<KeyPoint> kp1, kp2;
	Mat des1, des2;
	
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);


	detector->detect(img_temp, kp1);
	detector->detect(img_comp, kp2);

	descriptor->compute(img_temp, kp1, des1);
	descriptor->compute(img_comp, kp2, des2);
	

	vector<vector<DMatch> > knn_matches;
	matcher->knnMatch(des1, des2,knn_matches,2);

	
	
	const float ratio_thresh = 0.75f;

	vector< DMatch > good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}
	

	vector<Point2f> obj;
	vector<Point2f> scene;
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		obj.push_back(kp1[good_matches[i].queryIdx].pt);
		scene.push_back(kp2[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, RANSAC);
	vector<Point2f> obj_corners(4);
	obj_corners[0] = Point2f(0, 0);
	obj_corners[1] = Point2f((float)img_temp.cols, 0);
	obj_corners[2] = Point2f((float)img_temp.cols, (float)img_temp.rows);
	obj_corners[3] = Point2f(0, (float)img_temp.rows);
	vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H);


}
int main() {
	Mat img_temp_normal = imread("images/flag.png");
	Mat img_temp_rotated = imread("images/flag.png");
	Mat img_orig = imread("images/marioo.png");

	templateMatchingOpencvPreparedFunction(img_temp_normal, img_orig); 
	templateMatchingEqual(img_temp_normal, img_orig); 
	templateMatchingMean(img_temp_normal, img_orig);
	//I could not handle with this, if you run this you gonna take error. Maybe you want to examine this.
	/*featureMatchingSirf(img_temp_normal, img_orig);*/
	

}

