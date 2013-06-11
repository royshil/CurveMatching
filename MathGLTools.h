//
//  MathGLTools.h
//  CurveMatching
//
//  Created by roy_shilkrot on 5/23/13.
//
//
#pragma once

#ifndef CurveMatching_MathGLTools_h
#define CurveMatching_MathGLTools_h

#ifdef HAVE_MATHGL
#include <mgl2/mgl.h>
//#include <mgl2/window.h>
#endif
#include <algorithm>

#ifdef HAVE_MATHGL
#ifndef mgl_id
static int mgl_id = 0;
#endif
#endif

template<typename T>
void ShowMathGLDataAsHist(const vector<T>& values, const int* cutoffLine = NULL, const char* name = NULL) {
#ifdef HAVE_MATHGL
    vector<T> sorted(values);
    std::sort(sorted.rbegin(), sorted.rend());
//    cout << Mat_<T>(sorted) << endl;
    
    mglGraph mgl_gr;
    mglData mgl_x;
    mgl_x.Set(&(sorted[0]),sorted.size());
//    mglData mgl_h = mgl_gr.Hist(mgl_x);
    
    mgl_gr.SetRange('y', 0, 1);
    mgl_gr.SetRange('x', 0, sorted.size());
    mgl_gr.Box();
    mgl_gr.Axis();
    mgl_gr.Bars(mgl_x);
    if(cutoffLine != NULL)
        mgl_gr.Line(mglPoint(*cutoffLine,0), mglPoint(*cutoffLine,1));
    
    Mat img(mgl_gr.GetHeight(),mgl_gr.GetWidth(),CV_8UC3,(void*)mgl_gr.GetRGB());
    stringstream ss1;
    if(name == NULL) {
        ss1 << "MathGLDataAsHist " << mgl_id;
        mgl_id++;
    }
    else
        ss1 << name;
	imshow(ss1.str(), img);
#endif
}

template<typename T>
void ShowMathGLCompareCurves(const vector<Point_<T> >& a_canon, const vector<Point_<T> >& b_canon, const vector<double>& a_sig, const vector<double>& b_sig, double cc) {
#ifdef HAVE_MATHGL
    mglGraph gr;
    gr.SubPlot(2, 1, 0, "");
    
    vector<double> a_canon_x,a_canon_y;
    PolyLineSplit(a_canon, a_canon_x, a_canon_y);
    vector<double> b_canon_x,b_canon_y;
    PolyLineSplit(b_canon, b_canon_x, b_canon_y);
    
    mglData mgl_a_x(&(a_canon_x[0]),a_canon_x.size()),mgl_a_y(&(a_canon_y[0]),a_canon_y.size());
    mglData mgl_b_x(&(b_canon_x[0]),b_canon_x.size()),mgl_b_y(&(b_canon_y[0]),b_canon_y.size());
    
    gr.Title("Canonical");
    gr.Aspect(1, 1);
    gr.SetRanges(-.5, .5, -.5, .5);
    gr.Axis();
    gr.Grid();
    gr.Plot(mgl_a_x,mgl_a_y);
    gr.Plot(mgl_b_x,mgl_b_y);
    
    
    gr.SubPlot(2, 1, 1, "");
    mglData x(&(a_sig[0]),a_sig.size()),x1(&(b_sig[0]),b_sig.size());
    
    gr.Title("Signature");
    gr.SetRanges(0, max(a_sig.size(),b_sig.size()), 0, 0.55);
    gr.Axis();
    gr.Grid();
    gr.Plot(x);
    gr.Plot(x1);
    
    Mat img(gr.GetHeight(),gr.GetWidth(),CV_8UC3,(void*)gr.GetRGB());
    stringstream ss; ss << "cross correlation " << cc;
    putText(img, ss.str(), cv::Point(10,20), CV_FONT_NORMAL, 1.0, Scalar(255), 2);
    imshow("tmp", img);
    waitKey();
#endif
}

template<typename T>
void ShowMathGLCurve(const vector<Point_<T> > a_canon, const std::string& title)
{
#ifdef HAVE_MATHGL
	mglGraph gr;
	
	vector<double> a_canon_x,a_canon_y;
	PolyLineSplit(a_canon, a_canon_x, a_canon_y);
	
	mglData mgl_a_x(&(a_canon_x[0]),a_canon_x.size()),mgl_a_y(&(a_canon_y[0]),a_canon_y.size());
	
	gr.Title(title.c_str());
	gr.Aspect(1, 1);
	double axmin,axmax,aymin,aymax;
	minMaxIdx(a_canon_x, &axmin, &axmax);
	minMaxIdx(a_canon_y, &aymin, &aymax);
	gr.SetRanges(axmin, axmax, aymin, aymax);
	gr.Axis();
	gr.Grid();
	gr.Plot(mgl_a_x,mgl_a_y);
	
	Mat img(gr.GetHeight(),gr.GetWidth(),CV_8UC3,(void*)gr.GetRGB());
	stringstream ss1; ss1 << "MathMLCurves " << mgl_id; mgl_id++;
	imshow(ss1.str(), img);
#endif
}

template<typename T>
void ShowMathGLCurve(const vector<T> a_canon, const std::string& title)
{
	vector<T> count_; for(int x=0;x<a_canon.size();x++) count_.push_back(x);
	vector<Point2d> a_p2d; PolyLineMerge(a_p2d, count_, a_canon);
	ShowMathGLCurve(a_p2d,title);
}

template<typename T>
void ShowMathGLCurves(const vector<Point_<T> > a_canon, const vector<Point_<T> >& b_canon, const std::string& title)
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
	putText(img, ss.str(), cv::Point(10,20), CV_FONT_NORMAL, 1.0, Scalar(255), 2);
	stringstream ss1; ss1 << "MathMLCurves " << mgl_id; mgl_id++;
	imshow(ss1.str(), img);
#endif
}

template<typename T>
void ShowMathGLCurves(const vector<T> a_canon, const vector<T>& b_canon, const std::string& title)
{
	assert(a_canon.size() == b_canon.size());
	vector<double> count_; for(int x=0;x<a_canon.size();x++) count_.push_back(x);
	vector<Point2d> a_p2d; PolyLineMerge(a_p2d, count_, a_canon);
	vector<Point2d> b_p2d; PolyLineMerge(b_p2d, count_, b_canon);
	ShowMathGLCurves(a_p2d,
					 b_p2d,
					 title);
}

#endif
