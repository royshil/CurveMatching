/*
 *  CurveSignature.h
 *  CurveMatching
 *
 *  Created by Roy Shilkrot on 12/7/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */

#ifdef HAVE_MATHGL
#include <mgl2/mgl.h>
#include <mgl2/window.h>
#endif

#include <opencv2/features2d/features2d.hpp>

#pragma mark Utilities

#define CV_PROFILE(msg,code)	\
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

template<typename V>
Mat_<double> Find2DRigidTransform(const vector<Point_<V> >& a, const vector<Point_<V> >& b) {	
	//use PCA to find relational scale
	Mat a_ = Mat(a).reshape(1);
	Mat b_ = Mat(b).reshape(1);
	PCA a_pca(a_,Mat(),CV_PCA_DATA_AS_ROW), b_pca(b_,Mat(),CV_PCA_DATA_AS_ROW);
	double s = sqrt(b_pca.eigenvalues.at<V>(0)) / sqrt(a_pca.eigenvalues.at<V>(0));
	//	cout << a_pca.eigenvectors << endl << a_pca.eigenvalues << endl << a_pca.mean << endl;
	//	cout << b_pca.eigenvectors << endl << b_pca.eigenvalues << endl << b_pca.mean << endl;
	
	//convert to matrices and subtract mean
	Mat_<double> P(a.size(),2),Q(b.size(),2);
	Scalar a_m = mean(Mat(a)), b_m = mean(Mat(b));
	for (int i=0; i<a.size(); i++) { P(i,0) = a[i].x - a_m[0]; P(i,1) = a[i].y - a_m[1]; }
	for (int i=0; i<b.size(); i++) { Q(i,0) = b[i].x - b_m[0]; Q(i,1) = b[i].y - b_m[1]; }
	
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
	vector<Point> curve;
	ConvertCurve(curve_, curve);
	for (int i=0; i<curve.size(); i++) {
		circle(img, curve[i], 3, color, thickness);
	}
}

void GetCurveForImage(const Mat& filename, vector<Point>& curve, bool onlyUpper = true);

template<int x, int y>
void imshow_(const std::string& str, const Mat& img) {
	Mat big; resize(img,big,Size(x,y));
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

static int mgl_id = 0;
template<typename T>
void ShowMathMLCurves(const vector<Point_<T> > a_canon, const vector<Point_<T> >& b_canon, const std::string& title)
{
#ifdef HAVE_MATHGL
	mglGraph gr;
	
	vector<double> a_canon_x,a_canon_y;
	PolyLineSplit(a_canon, a_canon_x, a_canon_y);
	vector<double> b_canon_x,b_canon_y;
	PolyLineSplit(b_canon, b_canon_x, b_canon_y);
	
	mglData mgl_a_x(&(a_canon_x[0]),a_canon_x.size()),mgl_a_y(&(a_canon_y[0]),a_canon_y.size());
	mglData mgl_b_x(&(b_canon_x[0]),b_canon_x.size()),mgl_b_y(&(b_canon_y[0]),b_canon_y.size());
	
	gr.Title(title.c_str());
	gr.Aspect(1, 1);	
	double axmin,axmax,aymin,aymax,bxmin,bxmax,bymin,bymax;
	minMaxIdx(a_canon_x, &axmin, &axmax);
	minMaxIdx(a_canon_y, &aymin, &aymax);
	minMaxIdx(b_canon_x, &bxmin, &bxmax);
	minMaxIdx(b_canon_y, &bymin, &bymax);
	gr.SetRanges(min(axmin,bxmin), max(axmax,bxmax), min(aymin,bymin), max(aymax, bymax));
	gr.Axis(); 
	gr.Grid();
	gr.Plot(mgl_a_x,mgl_a_y);
	gr.Plot(mgl_b_x,mgl_b_y);	
	
	Mat img(gr.GetHeight(),gr.GetWidth(),CV_8UC3,(void*)gr.GetRGB());
	double cc = CalcCrossCorrelation(a_canon_y, b_canon_y);
	stringstream ss; ss << "cross correlation " << cc;
	putText(img, ss.str(), Point(10,20), CV_FONT_NORMAL, 1.0, Scalar(255), 2);
	stringstream ss1; ss1 << "MathMLCurves " << mgl_id; mgl_id++;
	imshow(ss1.str(), img);
#endif
}	

template<typename T>
void ShowMathMLCurves(const vector<T> a_canon, const vector<T>& b_canon, const std::string& title)
{
	assert(a_canon.size() == b_canon.size());
	vector<double> count_; for(int x=0;x<a_canon.size();x++) count_.push_back(x);
	vector<Point2d> a_p2d; PolyLineMerge(a_p2d, count_, a_canon);
	vector<Point2d> b_p2d; PolyLineMerge(b_p2d, count_, b_canon);
	ShowMathMLCurves(a_p2d,
					 b_p2d,
					 title);
}

#pragma mark Signatures Database

void PrepareSignatureDB(const vector<Point2d>& curve, vector<vector<double> >& DB, vector<Point>& DB_params);

template<typename T>
void PrepareSignatureDB(const vector<Point_<T> >& curve, vector<vector<double> >& DB, vector<Point>& DB_params) {
	vector<Point2d> curved; ConvertCurve(curve, curved);
	PrepareSignatureDB(curved,DB,DB_params);
}

void CompareCurvesUsingFLANN(const vector<Mat>& DB, 
							 const vector<vector<double> >& query_DB,
							 int& a_id,
							 int& a_subset_id,
							 int& b_subset_id);

void CompareCurvesUsingSignatureDBMatcher(FlannBasedMatcher& matcher,
										  const vector<Point>& typical_params,
										  const vector<vector<double> >& b_DB,
										  int& a_id,
										  int& a_len,
										  int& a_off,
										  int& b_len,
										  int& b_off,
										  double& score
										  );

void CompareCurvesUsingSignatureDB(const vector<Point>& a_DB_params,
								   const vector<Point>& b_DB_params,
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