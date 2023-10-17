#include <iostream>
#include <set>
#include <iterator>
#include <vector>

#include  <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sys/timeb.h>
#include "image_search.h"
#include "transformations.h"


using namespace std;
using namespace cv;


static vector<Vec2i> gcp_find(Mat img, vector<Mat> gcps) {
	vector<Vec2i> points;
	Mat temp;
	double minVal, maxVal;
	Point minLoc, maxLoc;
	for (int i = 0; i < gcps.size(); i++) {
		matchTemplate(img, gcps[i], temp, TM_CCOEFF_NORMED);
		
		minMaxLoc(temp, &minVal, &maxVal,&minLoc, &maxLoc);

		points.push_back( maxLoc);
	}

	return points;
}

int main() {

	Mat img = imread("C:\\Users\\vicin\\Desktop\\ImgProc\\transformed.png");
	namedWindow("img", WINDOW_AUTOSIZE);
	namedWindow("affine", WINDOW_AUTOSIZE);
	namedWindow("rfm", WINDOW_AUTOSIZE);

	struct timeb start, end;

	vector<Mat> gcps;
	char buf[256];
	for (int i = 0; i < 6; i++) {
		
		sprintf_s(buf, "C:\\Users\\vicin\\Desktop\\ImgProc\\gcp_%d.png", i);
		
		gcps.push_back(imread(buf));
		//GaussianBlur(gcps[i], gcps[i], Size(3, 3), 1.5);
	}
	Vec2i* gcp_coords;

	ftime(&start);

	gcp_coords = gcp_search(img, gcps, 0xF8, 14, 10);

	vector<Vec2i> gcps_transformed = vector<Vec2i>(gcp_coords, gcp_coords+6);

	//vector<Vec2i> gcps_transformed = gcp_find(img, gcps);

	//for (auto it = gcps_transformed.begin(); it != gcps_transformed.cend(); it++)
	//	cout <<*it << endl;




	
	Scalar_<int> cols[] = {
		{0,0,255},
		{0,255,0},
		{255,0,0},
		{255,255,255},
		{150,0,150},
		{0,150,150}
	};

	for (int i = 0; i < 6; i++) {
		cout << gcps_transformed[i] << endl;
		circle(img, gcps_transformed[i], 12, cols[i], 3);
	}

	imshow("img", img);

	vector<Vec2i> gcps_orig;
	ifstream f;
	f.open("C:\\Users\\vicin\\Desktop\\ImgProc\\metadata.txt");
	Vec2i temp;
	char line[256];
	for (int i = 0; i < 6; i++) {
		f.getline(line, 256);
		sscanf_s(line, "GCP %d -> Coords center:  (%d, %d)", &i, &temp[1], &temp[0]);

		gcps_orig.push_back(temp);
	}
	ftime(&end);

	cout << 1000 * (end.time - start.time) + (end.millitm - start.millitm) << endl;

	ftime(&start);
	
	vector<float> params = evaluateParamsAffine(gcps_orig, gcps_transformed);


	function<int(Vec2i)> lx = affine_x_transform(params);
	function<int(Vec2i)> ly = affine_y_transform(params);

	Mat out_affine =transformImage(img.clone(), lx, ly);

	ftime(&end);

	cout << 1000 * (end.time - start.time) + (end.millitm - start.millitm) << endl;
	imshow("affine", out_affine);

	ftime(&start);

	vector<float> params_rfm = evaluateParamsRFM(gcps_orig, gcps_transformed);


	function<int(Vec2i)> l2x = rfm_x_transform(params_rfm);
	function<int(Vec2i)> l2y = rfm_y_transform(params_rfm);

	Mat out_rfm = transformImage(img.clone(), l2x, l2y);

	ftime(&end);

	cout << 1000 * (end.time - start.time) + (end.millitm - start.millitm) << endl;
	imshow("rfm", out_rfm);



	waitKey(0);
	


	destroyAllWindows();
	//free(gcp_coords);
	return 0;
}