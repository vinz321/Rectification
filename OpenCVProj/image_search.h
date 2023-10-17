#pragma once

#include <iostream>
#include <set>
#include <vector>
#include <list>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

set<uint> approx_palette(Mat img, uint mask);
vector<set<uint>> generate_palette_matrix(Mat img, uint mask, int x_div, int y_div);
Vec2i* gcp_search(Mat img, vector<Mat>gcps, uint mask, int x_div, int y_div);
void subsearch(Mat img, Mat gcp, float* error, Vec2i* res, uint chunk_x, uint chunk_y);