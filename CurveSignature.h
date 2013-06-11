/*
 *  CurveSignature.h
 *  CurveMatching
 *
 *  Created by Roy Shilkrot on 12/7/12.
 *  Copyright (c) 2013 MIT
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 */


#include "opencv2/features2d/features2d.hpp"
#include <sys/stat.h>
#include "MathGLTools.h"

#pragma mark Utilities

#ifdef ENABLE_PROFILE
#define CV_PROFILE_MSG(msg,code)	\
{\
std::cout << msg << " ";\
double __time_in_ticks = (double)cv::getTickCount();\
{ code }\
std::cout << "DONE " << ((double)cv::getTickCount() - __time_in_ticks)/cv::getTickFrequency() << "s" << std::endl;\
}

#define CV_PROFILE(code)	\
{\
std::cout << #code << " ";\
double __time_in_ticks = (double)cv::getTickCount();\
{ code }\
std::cout << "DONE " << ((double)cv::getTickCount() - __time_in_ticks)/cv::getTickFrequency() << "s" << std::endl;\
}
#else
#define CV_PROFILE_MSG(msg,code) code
#define CV_PROFILE(code) code
#endif

bool fileExists(const std::string& filename);

template<typename T>
int closestPointOnCurveToPoint(const vector<cv::Point_<T> >& _tmp, const cv::Point& checkPt, const T cutoff) {
    vector<cv::Point_<T> > tmp = _tmp;
    Mat(tmp) -= Scalar(checkPt.x,checkPt.y);
    vector<float> tmpx,tmpy,tmpmag;
    PolyLineSplit(tmp, tmpx, tmpy);
    magnitude(tmpx,tmpy,tmpmag);
    double minDist = -1;
    cv::Point minLoc; minMaxLoc(tmpmag, &minDist,0,&minLoc);
    if(minDist<cutoff)
        return minLoc.x;
    else
        return -1;
}

template<typename T>
void saveCurveToFile(const vector<Point_<T> >& curve) {
	static int curve_id = 0;
	
	stringstream ss; ss << "curves/curves_"<<(curve_id++)<<".txt";
	while(fileExists(ss.str())) {
		ss.str("");
		ss << "curves/curves_"<<(curve_id++)<<".txt";
	}
	
	ofstream ofs(ss.str().c_str());
	ofs << curve.size() << "\n";
	for (int i=0; i<curve.size(); i++) {
		ofs << curve[i].x << " " << curve[i].y << "\n";
	}
	cout << "saved " << ss.str() << "\n";
}

template<typename T>
vector<Point_<T> > loadCurveFromFile(const string& filename) {
	vector<Point_<T> > curve;
	ifstream ifs(filename.c_str());
	int curve_size; ifs >> skipws >> curve_size;
	while (!ifs.eof()) {
		T x,y;
		ifs >> x >> y;
		curve.push_back(Point_<T>(x,y));
	}
	return curve;
}


template<typename V>
Mat_<double> Find2DRigidTransform(const vector<Point_<V> >& a, const vector<Point_<V> >& b, 
								  Point_<V>* diff = 0, V* angle = 0, V* scale = 0) {	
	//use PCA to find relational scale
	Mat_<V> P; Mat(a).reshape(1,a.size()).copyTo(P);
	Mat_<V> Q; Mat(b).reshape(1,b.size()).copyTo(Q);
	PCA a_pca(P,Mat(),CV_PCA_DATA_AS_ROW), b_pca(Q,Mat(),CV_PCA_DATA_AS_ROW);
	double s = sqrt(b_pca.eigenvalues.at<V>(0)) / sqrt(a_pca.eigenvalues.at<V>(0));
	//	cout << a_pca.eigenvectors << endl << a_pca.eigenvalues << endl << a_pca.mean << endl;
	//	cout << b_pca.eigenvectors << endl << b_pca.eigenvalues << endl << b_pca.mean << endl;
	
	//convert to matrices and subtract mean
//	Mat_<double> P(a.size(),2),Q(b.size(),2);
	Scalar a_m = Scalar(a_pca.mean.at<V>(0),a_pca.mean.at<V>(1));
    Scalar b_m = Scalar(b_pca.mean.at<V>(0),b_pca.mean.at<V>(1));
//	for (int i=0; i<a.size(); i++) { P(i,0) = a[i].x - a_m[0]; P(i,1) = a[i].y - a_m[1]; }
//	for (int i=0; i<b.size(); i++) { Q(i,0) = b[i].x - b_m[0]; Q(i,1) = b[i].y - b_m[1]; }
	P -= repeat((Mat_<V>(1,2) << a_m[0],a_m[1]), P.rows, 1);
	Q -= repeat((Mat_<V>(1,2) << b_m[0],b_m[1]), Q.rows, 1);
    
//    cout << "new mean for a " << mean(P) << "\n";
    
	//from http://en.wikipedia.org/wiki/Kabsch_algorithm
	Mat_<double> A = P.t() * Q;
	SVD svd(A);
	Mat_<double> C = svd.vt.t() * svd.u.t();
	double d = (determinant(C) > 0) ? 1 : -1;
	Mat_<double> R = svd.vt.t() * (Mat_<double>(2,2) << 1,0, 0,d) * svd.u.t();
	Mat_<double> T = (Mat_<double>(3,3) << 1,0,b_m[0]/s, 0,1,b_m[1]/s, 0,0,1) *
					(Mat_<double>(3,3) << s,0,0, 0,s,0, 0,0,s) *
					(Mat_<double>(3,3) << R(0,0),R(0,1),0, R(1,0),R(1,1),0, 0,0,1) *
					(Mat_<double>(3,3) << 1,0,-a_m[0], 0,1,-a_m[1], 0,0,1)
					;
	if (diff!=NULL) {
		diff->x = b_m[0]-a_m[0];
		diff->y = b_m[1]-a_m[1];
	}
	if (angle!=NULL) {
		*angle = atan2(R(1,0),R(0,0));
	}
	if (scale!=NULL) {
		*scale = s;
	}
	return T(Range(0,2),Range::all());
}

template<typename T,typename V>
Mat_<T> ConvertToMat(const vector<vector<V> >& mnt_DB) {
	Mat_<T> mnt_DB_m(mnt_DB.size(),mnt_DB[0].size());
	for (int i=0; i<mnt_DB.size(); i++) {
		for (int j=0; j<mnt_DB[i].size(); j++) {
			mnt_DB_m(i,j) = (T)(mnt_DB[i][j]);
		}
	}
	return mnt_DB_m;
}

template<typename T>
void drawCurvePoints(Mat& img, const vector<Point_<T> >& curve_, const Scalar& color, int thickness) {
	vector<cv::Point> curve;
	ConvertCurve(curve_, curve);
	for (int i=0; i<curve.size(); i++) {
		circle(img, curve[i], 3, color, thickness);
	}
}

void GetCurveForImage(const Mat& filename, vector<cv::Point>& curve, bool onlyUpper = true, bool getLower = false);

template <typename T>
void GetCurveForImage(const Mat& filename, vector<Point_<T> >& curve, bool onlyUpper = true, bool getLower = false) {
    vector<cv::Point> curve_2i;
    GetCurveForImage(filename, curve_2i,onlyUpper,getLower);
    ConvertCurve(curve_2i, curve);
}

template<int x, int y>
void imshow_(const std::string& str, const Mat& img) {
	Mat big; resize(img,big,cv::Size(x,y),-1,-1,INTER_NEAREST);
	imshow(str,big);
}

template<typename T,typename V>
vector<Point_<V> > YnormalizedCurve(const vector<Point_<T> >& curve) {
	vector<T> curvex,curvey; PolyLineSplit(curve, curvex, curvey);
	double minval,maxval;
	minMaxIdx(curvey, &minval,&maxval);
	vector<Point_<V> > curveout;
	for (int i=0; i<curve.size(); i++) {
		curveout.push_back(Point_<V>(i,(curvey[i] - minval) / (maxval - minval)));
	}
	return curveout;
}



#pragma mark Curvature Extrema

template<typename T>
vector<pair<char,int> > CurvatureExtrema(const vector<Point_<T> >& curve, vector<Point_<T> >& smooth, double smoothing_factor  = 3.0, bool visualizeCurvature = false) {
	if (curve.size() <= 0) {
		return vector<pair<char,int> >();
	}
	vector<double> kappa1;
	ComputeCurveCSS(curve, kappa1, smooth, smoothing_factor, true);
	
	if(visualizeCurvature)
		ShowMathGLCurve(kappa1, "curvature");
	
	vector<pair<char,int> > stringrep;
	stringrep.push_back(make_pair('s', 0));
	double cutoff = 0.025;
	for (int i=1; i<kappa1.size()-1; i++) {
		//inflection points
//		if ((kappa1[i-1] > 0 && kappa1[i] < 0))
//		{
//			stringrep.push_back(make_pair('I',i));
//		}
//		if ((kappa1[i-1] < 0 && kappa1[i] > 0)) {
//			stringrep.push_back(make_pair('i',i));
//		}
		
		//extrema
		if (abs(kappa1[i]) < cutoff) continue;
		if (kappa1[i] > 0) {
			if (kappa1[i-1] < kappa1[i] && kappa1[i+1] < kappa1[i]) {
				stringrep.push_back(make_pair('X',i));
			}
//			if (kappa1[i-1] > kappa1[i] && kappa1[i+1] > kappa1[i]) {
//				stringrep.push_back(make_pair('N',i));
//			}
		} else {
			if (kappa1[i-1] > kappa1[i] && kappa1[i+1] > kappa1[i]) {
				stringrep.push_back(make_pair('x',i));
			}
//			if (kappa1[i-1] < kappa1[i] && kappa1[i+1] < kappa1[i]) {
//				stringrep.push_back(make_pair('n',i));
//			}
		}
	}
	stringrep.push_back(make_pair('S', kappa1.size()-1));
	
	return stringrep;
}	

template<typename T>
void VisualizeExtremaPoints(Mat& outout,
							const vector<Point_<T> >& smooth, 
							const vector<pair<char,int> >& stringrep)
{
	for (int i=0; i<stringrep.size(); i++) {
		if (stringrep[i].first == 'X') {
			circle(outout, smooth[stringrep[i].second], 3, Scalar(0,0,255), CV_FILLED);
		}
		if (stringrep[i].first == 'N') {
			circle(outout, smooth[stringrep[i].second], 3, Scalar(255,255,0), CV_FILLED);
		}
		if (stringrep[i].first == 'x') {
			circle(outout, smooth[stringrep[i].second], 3, Scalar(0,255,0), CV_FILLED);
		}
		if (stringrep[i].first == 'n') {
			circle(outout, smooth[stringrep[i].second], 3, Scalar(0,255,255), CV_FILLED);
		}
		if (stringrep[i].first == 'I') {
			circle(outout, smooth[stringrep[i].second], 3, Scalar(50,50,50), CV_FILLED);
		}
		if (stringrep[i].first == 'i') {
			circle(outout, smooth[stringrep[i].second], 3, Scalar(100,100,100), CV_FILLED);
		}
	}
}	

template<typename T>
void VisualizeExtrema(const Mat& src, 
					  const vector<Point_<T> >& smooth, 
					  const vector<pair<char,int> >& stringrep,
					  const string& in_winname = "") 
{
	static int win_id = 0;
	
	Mat outout; //(src.size(),CV_8UC3,Scalar::all(0));
	if (src.channels() == 1) {
		cvtColor(src, outout, CV_GRAY2BGR);
	} else {
		src.copyTo(outout);
	}
	drawOpenCurve(outout, smooth, Scalar(255), 2);
	VisualizeExtremaPoints(outout,smooth,stringrep);
	stringstream ss;
	for (int i=0; i<stringrep.size(); i++) {
		ss << stringrep[i].first;
	}
	putText(outout, ss.str(), cv::Point(10,src.rows-20), CV_FONT_NORMAL, 0.5, Scalar(0,0,255), 1);
	stringstream winname;
	if (in_winname == "") {
		winname << "output" << win_id++;
	} else {
		winname << in_winname;
	}

	imshow(winname.str(), outout);
}	

Mat_<double> GetSmithWatermanHMatrix(const vector<pair<char,int> >& a, const vector<pair<char,int> >& b);

double MatchSmithWaterman(const vector<pair<char,int> >& a, const vector<pair<char,int> >& b, vector<cv::Point>& matching);

template<typename T>
void VisualizeMatching(const Mat& src, 
					   const vector<Point_<T> >& a_p2d,
					   const vector<Point_<T> >& b_p2d,
					   const vector<pair<char,int> >& stringrep,
					   const vector<pair<char,int> >& stringrep1,
					   const vector<cv::Point>& matching
					   ) 
{
	if (matching.size() == 0) return;
	Mat outout(src.size(),CV_8UC3); outout.setTo(0);
	drawOpenCurve(outout, a_p2d, Scalar(255,0,0), 1);
	drawOpenCurve(outout, b_p2d, Scalar(0,255,0), 1);
	VisualizeExtremaPoints(outout,a_p2d,stringrep);
	VisualizeExtremaPoints(outout,b_p2d,stringrep1);
	
	for (vector<cv::Point>::const_iterator itr = matching.begin(); itr != matching.end() - 1; ++itr) {
        cv::Point matchp = *itr;
		int a_ctrl_pt = matchp.x;
		int a_ctrl_pt_curve_pt = stringrep[a_ctrl_pt].second;
		int b_ctrl_pt = matchp.y;
		int b_ctrl_pt_curve_pt = stringrep1[b_ctrl_pt].second;
		line(outout, a_p2d[a_ctrl_pt_curve_pt], b_p2d[b_ctrl_pt_curve_pt], Scalar(0,0,255), 1);
	}
	
	imshow("matching", outout);
}

#pragma mark Signatures Database

void PrepareSignatureDB(const vector<Point2d>& curve, vector<vector<double> >& DB, vector<cv::Point>& DB_params);

template<typename T>
void PrepareSignatureDB(const vector<Point_<T> >& curve, vector<vector<double> >& DB, vector<cv::Point>& DB_params) {
	vector<Point2d> curved; ConvertCurve(curve, curved);
	PrepareSignatureDB(curved,DB,DB_params);
}

void CompareCurvesUsingFLANN(const vector<Mat>& DB, 
							 const vector<vector<double> >& query_DB,
							 int& a_id,
							 int& a_subset_id,
							 int& b_subset_id);

void CompareCurvesUsingSignatureDBMatcher(FlannBasedMatcher& matcher,
										  const vector<cv::Point>& typical_params,
										  const vector<vector<double> >& b_DB,
										  int& a_id,
										  int& a_len,
										  int& a_off,
										  int& b_len,
										  int& b_off,
										  double& score
										  );

void CompareCurvesUsingSignatureDB(const vector<cv::Point>& a_DB_params,
								   const vector<cv::Point>& b_DB_params,
								   const vector<vector<double> >& a_DB,
								   const vector<vector<double> >& b_DB,
								   int& a_len,
								   int& a_off,
								   int& b_len,
								   int& b_off,
								   double& score
								   );

void CompareCurvesUsingSignatureDB(const vector<Point2d>& a, 
								   const vector<Point2d>& b,
								   int& a_len,
								   int& a_off,
								   int& b_len,
								   int& b_off,
								   double& score
								   );

template<typename T>
void CompareCurvesUsingSignatureDB(const vector<Point_<T> >& a, 
								   const vector<Point_<T> >& b,
								   int& a_len,
								   int& a_off,
								   int& b_len,
								   int& b_off,
								   double& score
								   ) {
	vector<Point2d> ad; ConvertCurve(a, ad);
	vector<Point2d> bd; ConvertCurve(b, bd);
	CompareCurvesUsingSignatureDB(ad,bd,a_len,a_off,b_len,b_off,score);
}

#pragma mark Heat Kernel Signature computing and matching

template<typename T, typename V>
void CurveSignature(const vector<Point_<T> >& curve_T, Mat_<V>& hks, int sigmas) {
	vector<Point2d> curved; ConvertCurve(curve_T, curved);
	vector<Point2d> curve; 
//	if(curved.size() != 200) 
//		ResampleCurve(curved, curve, 200);
//	else
		curve = curved;

	vector<Point2d> smooth;
	
	Mat_<Vec3b> img(sigmas*10,curve.size(),Vec3b(0,0,0));
	hks.create(sigmas*10,curve.size());
	for (int i=0; i<hks.rows; i++) {
		double sigma = 1.0 + ((double)i)*0.1;
		vector<double> kappa(hks.cols,0.0);
		ComputeCurveCSS(curve, kappa, smooth, sigma, true);
		Mat(Mat(kappa).t()).copyTo(hks.row(i));
	} 
	
//	imshow_<800,200>("HKS(x,sigma)",hks);
//	waitKey();
}

void CompareSignatures(const Mat_<double>& a, const Mat_<double>& b);

#ifndef WITHOUT_OPENCL
void CompareCurvesGPU(const vector<Point2d>& a, 
					  const vector<Point2d>& b, 
					  double& a_sigma, 
					  double& b_sigma,
					  int& a_offset,
					  int& b_offset,
					  int& len
					  );

template<typename T>
void CompareCurvesGPU(const vector<Point_<T> >& a_, 
					  const vector<Point_<T> >& b_, 
					  double& a_sigma, 
					  double& b_sigma,
					  int& a_offset,
					  int& b_offset,
					  int& len
					  )
{
	vector<Point2d> a; ConvertCurve(a_, a);
	vector<Point2d> b; ConvertCurve(b_, b);
	CompareCurvesGPU(a, b, 
					 a_sigma, b_sigma,
					 a_offset, b_offset,
					 len
					 );
}


void CompareSignaturesGPU(const Mat_<float>& a, const Mat_<float>& b, 
						  double& a_sigma, 
						  double& b_sigma,
						  int& a_offset,
						  int& b_offset,
						  int& len
						  );
#endif
