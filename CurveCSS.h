/*
 *  CurveCSS.h
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
 */
#pragma once

#pragma mark Curves Utilities
template<typename T, typename V>
void PolyLineSplit(const vector<Point_<T> >& pl,vector<V>& contourx, vector<V>& contoury) {
	contourx.resize(pl.size()); 
	contoury.resize(pl.size());
	
	for (int j=0; j<pl.size(); j++) 
	{ 
		contourx[j] = (V)(pl[j].x); 
		contoury[j] = (V)(pl[j].y); 
	}
}

template<typename T, typename V>
void PolyLineMerge(vector<Point_<T> >& pl, const vector<V>& contourx, const vector<V>& contoury) {
	assert(contourx.size()==contoury.size());
	pl.resize(contourx.size());
	for (int j=0; j<contourx.size(); j++) {
		pl[j].x = (T)(contourx[j]);
		pl[j].y = (T)(contoury[j]);
	}
}

template<typename T, typename V>
void ConvertCurve(const vector<Point_<T> >& curve, vector<Point_<V> >& output) {
	output.clear();
	for (int j=0; j<curve.size(); j++) {
		output.push_back(Point_<V>(curve[j].x,curve[j].y));
	}
}

void ResampleCurve(const vector<double>& curvex, const vector<double>& curvey,
				   vector<double>& resampleX, vector<double>& resampleY,
				   int N, bool isOpen = false
				   );

template<typename T, typename V>
void ResampleCurve(const vector<Point_<T> >& curve, vector<Point_<V> >& output,
				   int N, bool isOpen = false
				   )
{
	vector<double> curvex, curvey, resampleX, resampleY;
	PolyLineSplit(curve, curvex, curvey);
	ResampleCurve(curvex,curvey,resampleX,resampleY,N,isOpen);
	PolyLineMerge(output, resampleX, resampleY);
}

template<typename T>
void drawOpenCurve(Mat& img, const vector<Point_<T> >& curve, Scalar color, int thickness) {
	if (curve.size() <= 0) {
		return;
	}
	vector<cv::Point> curve2i;
	ConvertCurve(curve, curve2i);
	for (int i=0; i<curve2i.size()-1; i++) {
		line(img, curve2i[i], curve2i[i+1], color, thickness);
	}
}

template<typename T>
void fillCurve(Mat& img, const vector<Point_<T> >& curve, Scalar color) {
	vector<cv::Point> curve2i;
	ConvertCurve(curve, curve2i);
    cv::Point* pptr = &(curve2i[0]);
	const cv::Point** ppptr = (const cv::Point**)&(pptr);
	int psize = curve2i.size();
	fillPoly(img, ppptr, &psize, 1, color);
}

#pragma mark CSS Image

void ComputeCurveCSS(const vector<double>& curvex, 
					 const vector<double>& curvey, 
					 vector<double>& kappa, 
					 vector<double>& smoothX,vector<double>& smoothY,
					 double sigma = 1.0,
					 bool isOpen = false);

template<typename T>
void ComputeCurveCSS(const vector<Point_<T> >& curve, 
					 vector<double>& kappa, 
					 vector<Point_<T> >& smooth,
					 double sigma,
					 bool isOpen = false
					 ) 
{
	vector<double> contourx(curve.size()),contoury(curve.size());
	PolyLineSplit(curve, contourx, contoury);
	
	vector<double> smoothx, smoothy;
	ComputeCurveCSS(contourx, contoury, kappa, smoothx, smoothy, sigma, isOpen);
	
	PolyLineMerge(smooth, smoothx, smoothy);	
}

vector<int> FindCSSInterestPoints(const vector<double>& kappa);

vector<int> ComputeCSSImageMaximas(const vector<double>& contourx_, const vector<double>& contoury_,
								   vector<double>& contourx, vector<double>& contoury, bool isClosedCurve = true);

void SmoothCurve(const vector<double>& curvex, 
				 const vector<double>& curvey,
				 vector<double>& smoothX, 
				 vector<double>& smoothY,
				 vector<double>& X,
				 vector<double>& XX,
				 vector<double>& Y,
				 vector<double>& YY,
				 double sigma,
				 bool isOpen = false);

template<typename T, typename V>
void SimpleSmoothCurve(const vector<Point_<T> >& curve, 
				 vector<Point_<V> >& smooth, 
				 double sigma,
				 bool isOpen = false) {
	vector<double> contourx(curve.size()),contoury(curve.size());
	PolyLineSplit(curve, contourx, contoury);
	
	vector<double> smoothx, smoothy, X,XX,Y,YY;
	SmoothCurve(contourx, contoury, smoothx, smoothy, X,XX,Y,YY, sigma, isOpen);
	
	PolyLineMerge(smooth, smoothx, smoothy);		
}


#pragma mark Curve Segments

template<typename T, typename V>
void GetCurveSegments(const vector<Point_<T> >& curve, const vector<int>& interestPoints, vector<vector<Point_<V> > >& segments, bool closedCurve = true) {
	if (closedCurve) {
		segments.resize(interestPoints.size());
	} else {
		segments.resize(interestPoints.size()+1);
	}

	for (int i = (closedCurve)?0:1; i<segments.size()-1; i++) {
		int intpt_idx = (closedCurve)?i:i-1;
		segments[i].clear();
		for (int j=interestPoints[intpt_idx]; j<interestPoints[intpt_idx+1]; j++) {
			segments[i].push_back(Point_<V>(curve[j].x,curve[j].y));
		}
	}
	if (closedCurve) {
		//put in the segment that passes the 0th point
		segments.back().clear();
		for (int j=interestPoints.back(); j<curve.size(); j++) {
			segments.back().push_back(Point_<V>(curve[j].x,curve[j].y));
		}
		for (int j=0; j<interestPoints[0]; j++) {
			segments.back().push_back(Point_<V>(curve[j].x,curve[j].y));
		}
	} else {
		//put in the segment after the last point
		segments.back().clear();
		for (int j=interestPoints.back(); j<curve.size(); j++) {
			segments.back().push_back(Point_<V>(curve[j].x,curve[j].y));
		}
		//put in the segment before the 1st point
		segments.front().clear();
		for (int j=0; j<interestPoints[0]; j++) {
			segments.front().push_back(Point_<V>(curve[j].x,curve[j].y));
		}
	}
	for (int i=0; i<segments.size(); i++) {
		vector<double> x,y;
		cout <<"segments[i].size() " << segments[i].size() << endl;
		PolyLineSplit(segments[i], x, y); ResampleCurve(x, y, x, y, 50,true); PolyLineMerge(segments[i], x, y);
	}
}
template<typename T, typename V>
void GetCurveSegmentsWithCSSImage(vector<Point_<T> >& curve, vector<int>& interestPoints, vector<vector<Point_<V> > >& segments, bool closedCurve = true) {
	vector<double> contourx(curve.size()),contoury(curve.size());
	PolyLineSplit(curve, contourx, contoury);
	
	vector<double> smoothx, smoothy;
	interestPoints = ComputeCSSImageMaximas(contourx, contoury, smoothx, smoothy);
	
	PolyLineMerge(curve, smoothx, smoothy);
	
	double minx,maxx; minMaxLoc(smoothx, &minx, &maxx);
	double miny,maxy; minMaxLoc(smoothy, &miny, &maxy);
	Mat drawing(maxy,maxx,CV_8UC3,Scalar(0));
	RNG rng(time(NULL));
	Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
//	vector<vector<Point_<T> > > contours(1,curve);
//	drawContours( drawing, contours, 0, color, 2, 8);
	drawOpenCurve(drawing, curve, color, 2);
	
	for (int m=0; m<interestPoints.size() ; m++) {
		circle(drawing, curve[interestPoints[m]], 5, Scalar(0,255), CV_FILLED);
	}
	imshow("curve interests", drawing);
	waitKey();
	
	GetCurveSegments(curve, interestPoints, segments, closedCurve);
}

#pragma mark Matching

double CalcCrossCorrelation(const vector<double>& x, const vector<double>& y);
template<typename T>
double CalcCrossCorrelation(const Mat_<T>& x, const Mat_<T>& y) {
	vector<double> xv;
	if (x.rows == 1) x.convertTo(xv, CV_64F);
	else if (x.cols == 1) Mat(x.t()).convertTo(xv, CV_64F);
	vector<double> yv;
	if (y.rows == 1) y.convertTo(yv, CV_64F);
	else if (y.cols == 1) Mat(y.t()).convertTo(yv, CV_64F);
	return CalcCrossCorrelation(xv,yv);
}


double MatchTwoSegments(const vector<Point2d>& a, const vector<Point2d>& b);
double MatchCurvesSmithWaterman(const vector<vector<Point2d> >& a, const vector<vector<Point2d> >& b, vector<cv::Point>& traceback);
double AdaptedMatchCurvesSmithWaterman(const vector<vector<Point2d> >& a, const vector<vector<Point2d> >& b, vector<cv::Point>& traceback);
