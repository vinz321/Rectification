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


// This function uses the OpenCV library to find the GCPs on the image

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

int main(int argc, char *argv[]) {
	if (argc < 2) {
		return 0;
	}

	char *folder = argv[1];
	char buf[256];
	sprintf_s(buf, "%s\\transformed.png", folder);
	Mat img = imread(buf);
	namedWindow("img", WINDOW_AUTOSIZE);
	namedWindow("affine", WINDOW_AUTOSIZE);
	namedWindow("rfm", WINDOW_AUTOSIZE);

	struct timeb start, end;

	vector<Mat> gcps;
	for (int i = 0; i < 6; i++) {
		
		sprintf_s(buf, "%s\\gcp_%d.png", folder ,i);
		
		gcps.push_back(imread(buf));
	}
	Vec2i* gcp_coords;

	ftime(&start);

	//gcp_coords = gcp_search(img, gcps, 0xF8, 14, 10);

	//vector<Vec2i> gcp_source = vector<Vec2i>(gcp_coords, gcp_coords+6);

	vector<Vec2i> gcp_source = gcp_find(img, gcps);

	//for (auto it = gcp_source.begin(); it != gcp_source.cend(); it++)
	//	cout <<*it << endl;




	
	Scalar_<int> cols[] = {
		{0,0,255},
		{0,255,0},
		{255,0,0},
		{255,255,255},
		{150,0,150},
		{0,150,150}
	};

	/*for (int i = 0; i < 6; i++) {
		cout << gcp_source[i] << endl;
		circle(img, gcp_source[i], 12, cols[i], 3);
	}*/

	imshow("img", img);

	vector<Vec2i> gcp_destination;
	ifstream f;
	f.open("C:\\Users\\vicin\\Desktop\\ImgProc\\metadata.txt");
	Vec2i temp;
	char line[256];
	for (int i = 0; i < 6; i++) {
		f.getline(line, 256);
		sscanf_s(line, "GCP %d -> Coords center:  (%d, %d)", &i, &temp[1], &temp[0]);
		
		gcp_destination.push_back(temp);
	}
	ftime(&end);

	cout << 1000 * (end.time - start.time) + (end.millitm - start.millitm) << endl;

	ftime(&start);
	

	vector<float> params = evaluateParamsAffine(gcp_destination, gcp_source);


	function<int(Vec2i)> lx = affine_x_transform(params);
	function<int(Vec2i)> ly = affine_y_transform(params);

	Mat out_affine = transformImage(img.clone(), lx, ly);


	ftime(&end);

	cout << 1000 * (end.time - start.time) + (end.millitm - start.millitm) << endl;

	Vec2i center = Vec2i(lx(img.size().width / 2), ly(img.size().height/2) );
	cout << "center: " << Vec2i(img.size().width / 2, img.size().height-img.size().height / 2)<<endl;
	//cout <<"center affine:" << center << endl;
	imshow("affine", out_affine);

	
	ftime(&start);

	vector<float> params_rfm = evaluateParamsRFM(gcp_destination, gcp_source);


	function<int(Vec2i)> l2x = rfm_x_transform(params_rfm);
	function<int(Vec2i)> l2y = rfm_y_transform(params_rfm);

	Mat out_rfm = transformImage(img.clone(), l2x, l2y);

	ftime(&end);

	cout << 1000 * (end.time - start.time) + (end.millitm - start.millitm) << endl;

	center = Vec2i(l2x(img.size().width / 2), l2y(img.size().height /2) );
	//cout << "center rfm:" << center << endl;

	imshow("rfm", out_rfm);

	imwrite("C:\\Users\\vicin\\Desktop\\ImgProc\\result_affine.png", out_affine);
	imwrite("C:\\Users\\vicin\\Desktop\\ImgProc\\result_rfm.png", out_rfm);

	waitKey(0);
	


	destroyAllWindows();
	//free(gcp_coords);
	return 0;
}