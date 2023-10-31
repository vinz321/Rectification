#include "image_search.h"

static Point gcp_find(Mat img, Mat gcp, double *maxVal) {
	vector<Vec2i> points;
	Mat temp;
	double minVal;
	Point minLoc, maxLoc;
	matchTemplate(img, gcp, temp, TM_CCOEFF);

	minMaxLoc(temp, &minVal, maxVal, &minLoc, &maxLoc);

	return maxLoc;
}

set <uint> approx_palette(Mat img, uint mask, bool isGcp=false) {
	set <uint> palette = set<uint>();
	//GaussianBlur(img, img, Size(3, 3), 1.5);

	for (int y = 0; y < img.size().height; y++) {
		for (int x = 0; x < img.size().width; x++) {
			
			int res = 0;
			Vec3b v = img.at <Vec3b>(x, y);
			res |= (v[0] & mask);
			res <<= 8;
			res |= (v[1]&mask);
			res <<= 8;
			res |= (v[2] & mask);

			palette.insert(res);
		}
	}

	if (isGcp) {
		copyMakeBorder(img, img, 1, 1, 1, 1, BorderTypes::BORDER_REPLICATE);
		GaussianBlur(img, img, Size(3, 3), 1);
		//resize(img, img, { 20,20 });
		/*int kernel_arr[9] = { 255,255,255, 255,255,255, 255,255,255 };
		Mat kern = Mat(3, 3, CV_32F, kernel_arr);
		kern = kern / 9;
		filter2D(img, img, -1, kern);*/

		for (int y = 0; y < img.size().height; y++) {
			for (int x = 0; x < img.size().width; x++) {
				int res = 0;
				Vec3b v = img.at <Vec3b>(x, y);
				res |= (v[0] & mask);
				res <<= 8;
				res |= (v[1] & mask);
				res <<= 8;
				res |= (v[2] & mask);

				palette.insert(res);
			}
		}
	}

	return palette;
}

vector<set<uint>> generate_palette_matrix(Mat img, uint mask, int x_div, int y_div) {

	vector<set<uint>> array = vector<set<uint>>(x_div*y_div);
	int w = img.size().width, h = img.size().height;

	int x_off = img.size().width / x_div, y_off = img.size().height / y_div;
	set<uint> t;
	for (int y = 0; y < y_div; y ++) {
		for (int x = 0; x < x_div; x ++) {
			Rect imgRect = Rect({}, img.size());
			Rect roi = Rect(x * x_off, y * y_off, x_off + 10, y_off + 10);
			roi = imgRect & roi;
				/*t= approx_palette(img.colRange(Range(x * x_off, (x + 1) * x_off+10)).
				rowRange(Range(y * y_off, (y + 1) * y_off+10)).clone(), mask);*/


			t = approx_palette(img(roi).clone(), mask, false);
			array[y * x_div + x] = t;
		}
	}
	
	return array;
}

Vec2i* gcp_search(Mat img, vector<Mat> gcps, uint mask, int x_div, int y_div) {
	Vec2i* candidates = (Vec2i*)malloc(gcps.size() * sizeof(Vec3b));
	float* errors = (float*)malloc(gcps.size() * sizeof(float));
	vector<set<uint>> gcp_palettes = vector<set<uint>>(gcps.size());

	int x_off = img.size().width / x_div, y_off = img.size().height / y_div;

	int x, y, i = 0;
	int gcp_size = gcps.size();
	set<uint> s;
	for (auto it = gcps.begin(); it < gcps.cend(); it++) {
		s = approx_palette((*it).clone(), mask, true);
		//errors[i] = 2000;
		errors[i] = 0;

		gcp_palettes[i] = s;
		i++;
	}

	vector<set<uint>> palettes = generate_palette_matrix(img, mask, x_div, y_div);

	for (y = 0; y < y_div; y++) {
		for (x = 0; x < x_div; x++) {
			for (i = 0; i < gcp_size; i++) {
				if (!includes(palettes[y * x_div + x].begin(), palettes[y * x_div + x].end(),
					gcp_palettes[i].begin(), gcp_palettes[i].end()))
					continue;
				Rect imgRect = Rect({}, img.size());
				Rect roi = Rect(x * x_off, y * y_off, x_off + 10, y_off + 10);
				roi = imgRect & roi;

				subsearch(img(roi),
						gcps[i], errors + i, candidates + i, x * x_off, y * y_off);
				//cout << *(errors + i) << " " << i;
			}

			//cout << endl;
		}
	}

	free(errors);

	return candidates;
}


void subsearch(Mat img, Mat gcp, float* error, Vec2i* candidate, uint chunk_x, uint chunk_y) {
	int w = img.size().width, h = img.size().height;

	double temp_err;
	Point p = gcp_find(img, gcp, &temp_err);
	if (temp_err > *error) {
		*error = (float)temp_err;
		*candidate = Vec2i(p.x + chunk_x + 5, p.y + chunk_y + 5);
	}

	//for (int y = 0; y < h-10; y++) {
	//	for (int x = 0; x < w - 10; x++) {

	//		double temp_err;

	//		Point p=gcp_find(img(Rect(x, y, 10, 10)), gcp, &temp_err);
	//		//Mat roi = img(Rect(x, y, 10, 10));
	//		//Mat res;
	//		//absdiff(roi, gcp, res);
	//		////res = res.mul(res);
	//		//Scalar_<float> mean_ = (Scalar_<float>)mean(res);
	//		////cout << mean_<<endl;
	//		//float norm_ = (mean_[0]+mean_[1]+ mean_[2])/3;
	//		if (norm_ < *error) {
	//			*error = temp_err;
	//			
	//			*candidate = Vec2i(x+chunk_x+5, y+chunk_y+5);
	//		}
	//	}
	//}
}


