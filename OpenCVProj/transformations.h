#pragma once


#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;


vector<float> evaluateParamsAffine(vector<Vec2i> gcp_transformed, vector<Vec2i> gcp_orig);
vector<float> evaluateParamsRFM(vector<Vec2i> gcp_transformed, vector<Vec2i> gcp_orig);

function<int (Vec2i)> affine_x_transform(vector<float> params);
function<int (Vec2i)> affine_y_transform(vector<float> params);

function<int(Vec2i)> rfm_x_transform(vector<float> params);
function<int(Vec2i)> rfm_y_transform(vector<float> params);

Mat transformImage(Mat img, function<int(Vec2i)> lx, function<int(Vec2i)> ly);
