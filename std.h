/*
 *  std.h
 *  CurveMatching
 *
 *  Created by Roy Shilkrot on 11/28/12.
 *  Copyright (c) 2013 MIT
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 */
#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <set>
#include <fstream>
#include <iostream>
#include <limits>
#include <iomanip>
#include <numeric>

using namespace std;

#undef check

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <dirent.h>

bool hasEnding (std::string const &fullString, std::string const &ending);
bool hasEndingLower (string const &fullString_, string const &_ending);
void open_imgs_dir(const char* dir_name, std::vector<std::string>& images_names);
vector<string> open_dir(const char* dir_name, const char* filter);
vector<string> open_dir(const char* dir_name, std::vector<std::string>& filter);

#include "/usr/local/include/eigen3/Eigen/Eigen"

template<typename T1,typename T2>
inline bool sortbyfirst (pair<T1,T2> i,pair<T1,T2> j) { return (i.first<j.first); }

#ifndef SSTR
#define SSTR( x ) dynamic_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()
#endif
