#include "transformations.h"
#include <opencv2/core/hal/interface.h>


// Regression function to minimize the errors of the affine function relative to the gcps
static vector<float> regression(vector<float> &params, vector<Vec2i> gcp_destination, vector<Vec2i> gcp_source, float learn_rate) {
	Vec<float, 6> A = Vec<float, 6>(params.begin()._Ptr);
	
	Vec<float, 6> delta;
	Vec<float, 6> temp;

	float sqr_err=0;
	for(int j=0;j<gcp_destination.size();j++)
	{
		sqr_err = 0;
		delta = Vec<float, 6>();
		for (int i = 0; i < gcp_destination.size(); i++) {

			float X1 = (A[0] + A[1] * gcp_destination[i][0] + A[2] * gcp_destination[i][1] - gcp_source[i][0]);

			float X2 = (A[3] + A[4] * gcp_destination[i][0] + A[5] * gcp_destination[i][1] - gcp_source[i][1]);

			temp[0] = X1;
			temp[1] = gcp_destination[i][0] * X1;
			temp[2] = gcp_destination[i][1] * X1;

			temp[3] = X2;
			temp[4] = gcp_destination[i][0] * X2;
			temp[5] = gcp_destination[i][1] * X2;

			delta += temp;
			sqr_err += X1 * X1 + X2 * X2;
		}

		sqr_err /= 12;
		A = A - learn_rate*delta/12;
	}


	return vector<float>(&A[0], &A[6]);
}

// Retrieves the parameters for the affine transformation
vector<float> evaluateParamsAffine(vector<Vec2i> gcp_destination, vector<Vec2i> gcp_source) {
	Matx<float, 6, 6> mat;
	Vec<float, 6> sol;
	for (int i = 0; i < 3; i++) {
		mat(i, 0) = 1;
		mat(i,1) = gcp_destination[i][0];
		mat(i,2) = gcp_destination[i][1];

		sol[i] = gcp_source[i][0];

		mat(i+3, 3) = 1;
		mat(i+3, 4) = gcp_destination[i][0];
		mat(i+3, 5) = gcp_destination[i][1];
		
		sol[i+3] = gcp_source[i][1];
	}

	float *res = mat.solve(sol).val;
	vector<float> params = vector<float>(res, res + 6);
	

	return regression(params, gcp_destination, gcp_source,0.00003f);
	//return params;
}

// Retrieves the parameters for the Rational Function Model

vector<float> evaluateParamsRFM(vector<Vec2i> gcp_destination, vector<Vec2i> gcp_source) {
	Matx<float, 10, 10> mat;
	Vec<float, 10> sol;
	for (int i = 0; i < 5; i++) {
		mat(i, 0) = 1;
		mat(i, 1) = gcp_destination[i][0];
		mat(i, 2) = gcp_destination[i][1];
		mat(i, 3) = -(gcp_source[i][0] * gcp_destination[i][0]);
		mat(i,4) = -(gcp_source[i][0] * gcp_destination[i][1]);

		sol[i] = gcp_source[i][0];

		mat(i + 5, 5) = 1;
		mat(i + 5, 6) = gcp_destination[i][0];
		mat(i + 5, 7) = gcp_destination[i][1];
		mat(i + 5, 8) = -(gcp_source[i][1] * gcp_destination[i][0]);
		mat(i + 5, 9) = -(gcp_source[i][1] * gcp_destination[i][1]);

		sol[i + 5] = gcp_source[i][1];
	}

	float* res = mat.solve(sol).val;

	return vector<float>(res, res + 10);
}

// The 2 following functions generate the functions relative to parameters generated with the affine model 

function<int(Vec2i)> affine_x_transform(vector<float> params) {
	return [=](Vec2i v) {return (int)(params[0] + params[1] * v[0] + params[2] * v[1]); };
}

function<int(Vec2i)> affine_y_transform(vector<float> params) {
	return [=](Vec2i v) {return (int)(params[3] + params[4] * v[0] + params[5] * v[1]); };
}

// The 2 following functions generate the functions relative to parameters generated with the RFM 

function<int(Vec2i)> rfm_x_transform(vector<float> params) {
	return [=](Vec2i v) {return (int)((params[0]	+	params[1] * v[0]	+	params[2] * v[1]) / 
										(1	+	params[3]*v[0]	+	params[4]*v[1])); };
}

function<int(Vec2i)> rfm_y_transform(vector<float> params) {
	return [=](Vec2i v) {return (int)((params[5]	+	params[6] * v[0]	+	params[7] * v[1]) / 
										(1	+	params[8] * v[0]	+	params[9] * v[1])); };
}

//Takes in the functions to map the destination space to the source space

Mat transformImage(Mat img, function<int(Vec2i)> lx, function<int(Vec2i)> ly) {
	Mat res=Mat(Size(img.size().width, img.size().height), CV_8UC3);

	for (uint y = 0; y < res.size().height; y++) {
		for (uint x = 0; x < res.size().width; x++) {
			int new_x =lx(Vec2i(x, y)) , new_y=  ly(Vec2i(x, y));
			//cout << new_x << " " << new_y<<endl;
			//if (new_x >= 368 && new_x <= 370 && new_y >= 256 && new_y <= 258)
			//	cout << "center transform:" << Vec2i(x, y) << endl;
			//cout << pos << endl;
			
			if(new_x<img.size().width && new_x>=0 && new_y<img.size().height && new_y>=0)
				res.at<Vec3b>(Point2i(x,y)) = img.at<Vec3b>(Point2i(new_x,new_y));
		}
	}

	return res;
}




