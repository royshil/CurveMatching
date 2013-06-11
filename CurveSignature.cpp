/*
 *  CurveSignature.cpp
 *  CurveMatching
 *
 *  Created by Roy Shilkrot on 12/7/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */

#include "std.h"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/flann/dist.h>
using namespace cv;

#include "CurveCSS.h"
#include "CurveSignature.h"

#pragma mark Utilities

bool fileExists(const std::string& filename)
{
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1)
    {
        return true;
    }
    return false;
}

void GetCurveForImage(const Mat& filename, vector<Point>& whole, vector<Point>& curve_upper, vector<Point>& curve_lower) {
	assert(!filename.empty());
	Mat tmp; filename.copyTo(tmp);
	Mat gray; 
	if(tmp.type() == CV_8UC3)
		cvtColor(tmp, gray, CV_BGR2GRAY);
	else if(tmp.type() == CV_8UC1)
		gray = tmp;
	else 
		cvError(-1, "GetCurveForImage", "unsupported image format", __FILE__, __LINE__);


	threshold(gray, gray, 128, 255, THRESH_BINARY);
//	imshow("input",gray);
	
	vector<vector<Point> > contours;
	findContours( gray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	if (contours.size()<=0) return;

	vector<Point> upperCurve = contours[0];
	if (upperCurve.size() <= 50) {
		return;
	}
	
	//find minimal and maximal X coord
	vector<double> x,y;
	PolyLineSplit(contours[0], x, y);
	Point minxp,maxxp;
	minMaxLoc(x, 0, 0, &minxp, &maxxp);
	int minx = minxp.x,maxx = maxxp.x;
	if (minx > maxx) swap(minx, maxx);
	
	//take lower and upper halves of the curve
	vector<Point> upper,lower;
	upper.insert(upper.begin(),contours[0].begin()+minx,contours[0].begin()+maxx);
	lower.insert(lower.begin(),contours[0].begin()+maxx,contours[0].end());
	lower.insert(lower.end(),contours[0].begin(),contours[0].begin()+minx);
	
	//test which is really the upper part, by looking at the y-coord of the mid point
	
	if (lower[lower.size()/2].y <= upper[upper.size()/2].y) {
		curve_upper = lower;
		curve_lower = upper;
	} else {
		curve_upper = upper;
		curve_lower = lower;
	}
	
	//make sure it goes left-to-right
	if (curve_upper.front().x > curve_upper.back().x) { //hmmm, need to flip
		reverse(curve_upper.begin(), curve_upper.end());
	}		
	
	whole.clear();
	whole.insert(whole.begin(),curve_upper.rbegin(),curve_upper.rend());
	whole.insert(whole.begin(),curve_lower.begin(),curve_lower.end());
}	

void GetCurveForImage(const Mat& filename, vector<Point>& curve, bool onlyUpper, bool getLower) {
	vector<Point> whole,upper,lower;
	GetCurveForImage(filename,whole,upper,lower);
	if (onlyUpper) {
		if (getLower) 
			curve = lower;
		else
			curve = upper;
	} else {
		curve = whole;
	}
}


#pragma mark Signature Database extracting and matching

void PrepareSignatureDB(const vector<Point2d>& curve_, vector<vector<double> >& DB, vector<Point>& DB_params) {
	vector<Point2d> curve;
	if (curve_.size() != 200) {
		ResampleCurve(curve_, curve, 200, true);
	}else {
		curve = curve_;
	}

	
	vector<double> kappa; 
	vector<Point2d> smooth;
	SimpleSmoothCurve(curve, smooth, 5.0, true);
	vector<Point2d> small;
	
	DB.clear(); DB_params.clear();
	for (int len = 50; len < smooth.size() - 2; len+=5) {
		//iterate different curve sizes, starting at 20 points
//		cout << "len " << len <<  endl;
		
		for (int off = (smooth.size() - len); off >= 0; off-=5) {
			//iterate segments on A curve
			vector<Point2d> small_smooth_input(smooth.begin()+off,smooth.begin()+off+len);
			
			//resample to N points
			ResampleCurve(small_smooth_input, small, 200, true);
			
			//compute curvature
			vector<Point2d> small_smooth;
			ComputeCurveCSS(small, kappa, small_smooth, 0.66667, true);
			vector<double> kappa_(kappa.begin()+1,kappa.end()-1);
			
			DB.push_back(kappa_);
			DB_params.push_back(Point(len,off));
		}
	}
	
	cout << "DB size " << DB.size() << endl;
}



template<class T>
struct CrossCorrelationDistance
{
    typedef cvflann::True is_kdtree_distance;
    typedef cvflann::True is_vector_space_distance;
	
    typedef T ElementType;
    typedef typename Accumulator<T>::Type ResultType;
	
    template <typename Iterator1, typename Iterator2>
    ResultType operator()(Iterator1 x, Iterator2 y, size_t size, ResultType /*worst_dist*/ = -1) const
    {
        int i,n = size;
		ResultType mx,my,sx,sy,sxy,denom,r;
		
		/* Calculate the mean of the two series x[], y[] */
		mx = 0;
		my = 0;   
		for (i=0;i<n;i++) {
			mx += x[i];
			my += y[i];
		}
		mx /= n;
		my /= n;
		
		/* Calculate the denominator */
		sx = 0;
		sy = 0;
		for (i=0;i<n;i++) {
			sx += (x[i] - mx) * (x[i] - mx);
			sy += (y[i] - my) * (y[i] - my);
		}
		denom = sqrt(sx*sy);
		
		/* Calculate the correlation series */
		sxy = 0;
		for (i=0;i<n;i++) {
			sxy += (x[i] - mx) * (y[i] - my);
		}
		r = sxy / denom;
		
		return r;
		
    }
	
    template <typename U, typename V>
    inline ResultType accum_dist(const U& a, const V& b, int) const
    {
        return a*b;
    }
};



void CompareCurvesUsingFLANN(const vector<Mat>& DB, 
							 const vector<vector<double> >& query_DB,
							 int& a_id,
							 int& a_subset_id,
							 int& b_subset_id) 
{
	Mat packed(DB.size()*DB[0].rows,DB[0].cols,CV_64FC1);
	int rowstep = DB[0].rows;
	for (int i=0; i<DB.size(); i++) {
		DB[i].convertTo(packed(Range(i*rowstep,(i+1)*rowstep),Range::all()),CV_64F);
	}
	Mat packed_query(query_DB.size(),query_DB[0].size(),CV_64F);
	for (int i=0; i<query_DB.size(); i++) {
		Mat(Mat(query_DB[i]).t()).copyTo(packed_query.row(i));
	}
	
	cout << "training..";
	cv::flann::GenericIndex<cvflann::L2<double> > gi(packed,cvflann::KDTreeIndexParams());
	cout << "DONE\n";
	Mat indices(packed_query.rows,1,CV_32SC1),dists(packed_query.rows,1,CV_64FC1);
	CV_PROFILE(gi.knnSearch(packed_query, indices, dists, 1, cvflann::SearchParams());)
		
//	cout << dists << endl;
	
//	dists = Mat(abs(dists));
	
//	minMaxIdx(dists, 0, 0, &b_subset_id);
	vector<pair<double,int> > scores;
	for (int i=0; i<indices.rows; i++) {
		double d = dists.at<double>(i);
		if (isnan(d)) continue;
		scores.push_back(make_pair(d, i));
	}
	sort(scores.begin(),scores.end());
	for (int i=0; i<20; i++) {
		cout << scores[i].first << " = " << scores[i].second << "(" << indices.at<int>(scores[i].second) << ")" << endl;
	}
	b_subset_id = scores.front().second;
	
	a_id = indices.at<int>(b_subset_id) / rowstep;
	a_subset_id = indices.at<int>(b_subset_id) % rowstep;
	cout << "minimal "<< indices.at<int>(b_subset_id) <<" = " << a_id << "(" << a_subset_id << ") -> " << b_subset_id << ": " << dists.at<double>(b_subset_id) << endl;
		
	{
		vector<double> a_sig(packed.cols); 
		memcpy(&(a_sig[0]), packed.row(indices.at<int>(b_subset_id)).data, sizeof(double)*a_sig.size());
		ShowMathGLCurves(query_DB[b_subset_id], a_sig, "curvatures");
	}
}
	
void CompareCurvesUsingSignatureDBMatcher(FlannBasedMatcher& matcher,
										  const vector<Point>& typical_params,
										  const vector<vector<double> >& b_DB,
										  int& a_id,
										  int& a_len,
										  int& a_off,
										  int& b_len,
										  int& b_off,
										  double& score
										  ) 
{	
	std::vector< DMatch > matches;
	{
		Mat_<float> b_DB_m = ConvertToMat<float,double>(b_DB);
		CV_PROFILE(matcher.match( b_DB_m, matches );)
	}
	double min_dist = std::numeric_limits<double>::max();
	
	DMatch min_match;
	//-- Quick calculation of max and min distances between keypoints
	for( int i = matches.size()-1; i >=0 ; i-- )
	{ 
//		int imgidx = matches[i].imgIdx;
		double dist = (max(1.0,200.0 - (double)(typical_params[matches[i].queryIdx].x)) +
					   max(1.0,200.0 - (double)(typical_params[matches[i].trainIdx].x))) +  
						100.0*matches[i].distance;
		if( dist < min_dist ) { min_dist = dist; min_match = matches[i]; }
		//		if( dist > max_dist ) max_dist = dist;
	}
	
	a_id = min_match.imgIdx;
	//	printf("-- Max dist : %f \n", max_dist );
	printf("-- Min dist : %f, %d(%d,%d) -> %d(%d,%d) \n", 
		   min_dist, 
		   min_match.queryIdx, 
		   typical_params[min_match.queryIdx].x,
		   typical_params[min_match.queryIdx].y,
		   min_match.trainIdx,
		   typical_params[min_match.trainIdx].x,
		   typical_params[min_match.trainIdx].y
		   );
	
//	cout << Mat(a_DB[min_match.queryIdx]).t() << endl << Mat(b_DB[min_match.trainIdx]).t() << endl;
	
	a_len = typical_params[min_match.queryIdx].x;
	a_off = typical_params[min_match.queryIdx].y;
	b_len = typical_params[min_match.trainIdx].x;
	b_off = typical_params[min_match.trainIdx].y;
}	

void CompareCurvesUsingSignatureDB(const vector<Point>& a_DB_params,
								   const vector<Point>& b_DB_params,
								   const vector<vector<double> >& a_DB,
								   const vector<vector<double> >& b_DB,
								   vector<pair<double,DMatch> >& scores_to_matches
								   ) 
{	
	FlannBasedMatcher matcher;
//	BFMatcher matcher(NORM_L2);
	std::vector< DMatch > matches;
	{
		Mat_<float> mnt_DB_m = ConvertToMat<float,double>(a_DB);
		Mat_<float> obj_DB_m = ConvertToMat<float,double>(b_DB);
		vector<Mat> obj_DB_mv(1,obj_DB_m);
		matcher.add(obj_DB_mv);
		CV_PROFILE(matcher.train();)
		CV_PROFILE(matcher.match( mnt_DB_m, matches );)
	}
	
	DMatch min_match;
	
	vector<pair<double,int> > scores;
	for( int i = matches.size()-1; i >=0 ; i-- )
	{
		double d = //(max(1.0,200.0 - (double)(a_DB_params[matches[i].queryIdx].x)) + 
//					   max(1.0,200.0 - (double)(b_DB_params[matches[i].trainIdx].x))) +  
						10000.0 * matches[i].distance;
		if (isnan(d)) continue;
		scores.push_back(make_pair(d, i));
	}
	sort(scores.begin(),scores.end());
	
	scores_to_matches.clear();
	for (int i=0; i<200; i++) {
//		cout << scores[i].first << " = " << scores[i].second << endl;
		scores_to_matches.push_back(make_pair(scores[i].first, matches[scores[i].second]));
	}
	min_match = matches[scores.front().second];
//	double min_dist = scores.front().first;
	
	//	printf("-- Max dist : %f \n", max_dist );
//	printf("-- Min dist : %f, %d(%d,%d) -> %d(%d,%d) \n", 
//		   min_dist, 
//		   min_match.queryIdx, 
//		   a_DB_params[min_match.queryIdx].x,
//		   a_DB_params[min_match.queryIdx].y,
//		   min_match.trainIdx,
//		   b_DB_params[min_match.trainIdx].x,
//		   b_DB_params[min_match.trainIdx].y
//		   );
//	
//	cout << Mat(a_DB[min_match.queryIdx]).t() << endl << Mat(b_DB[min_match.trainIdx]).t() << endl;
//	{
//		vector<double> a_sig,b_sig;
//		a_sig = a_DB[matches[scores.front().second].queryIdx], 
//		b_sig = b_DB[matches[scores.front().second].trainIdx];
//		ShowMathMLCurves(a_sig, b_sig, "curvatures0");
//
//		a_sig = a_DB[matches[scores[1].second].queryIdx];
//		b_sig = b_DB[matches[scores[1].second].trainIdx];
//		ShowMathMLCurves(a_sig, b_sig, "curvatures1");
//
//		a_sig = a_DB[matches[scores[2].second].queryIdx];
//		b_sig = b_DB[matches[scores[2].second].trainIdx];
//		ShowMathMLCurves(a_sig, b_sig, "curvatures2");
//
//		a_sig = a_DB[matches[scores[3].second].queryIdx];
//		b_sig = b_DB[matches[scores[3].second].trainIdx];
//		ShowMathMLCurves(a_sig, b_sig, "curvatures3");
//		
//	}
}	

void CompareCurvesUsingSignatureDB(const vector<Point2d>& a, 
								   const vector<Point2d>& b,
								   int& a_len,
								   int& a_off,
								   int& b_len,
								   int& b_off,
								   double& score
								   ) 
{
	vector<Point> a_DB_params,b_DB_params;
	vector<vector<double> > a_DB,b_DB;
	PrepareSignatureDB(a, a_DB, a_DB_params);
	PrepareSignatureDB(b, b_DB, b_DB_params);
	
	vector<pair<double,DMatch> > scores_to_matches;
	CompareCurvesUsingSignatureDB(a_DB_params,b_DB_params,a_DB,b_DB,scores_to_matches);
	
	//re-rank results by RMSE measure after recovering rigid transformation
	for (int i=0; i<scores_to_matches.size(); i++) {
		int _a_len = a_DB_params[scores_to_matches[i].second.queryIdx].x;
		int _a_off = a_DB_params[scores_to_matches[i].second.queryIdx].y;
		int _b_len = b_DB_params[scores_to_matches[i].second.trainIdx].x;
		int _b_off = b_DB_params[scores_to_matches[i].second.trainIdx].y;
		
		vector<Point2d> a_subset(a.begin() + _a_off, a.begin() + _a_off + _a_len);
		vector<Point2d> b_subset(b.begin() + _b_off, b.begin() + _b_off + _b_len);
		
		ResampleCurve(a_subset, a_subset, 200, true);
		ResampleCurve(b_subset, b_subset, 200, true);
		
		Mat trans = Find2DRigidTransform(a_subset, b_subset);
//		cout << trans << endl;
		vector<Point2d> a_trans;
		cv::transform(a_subset,a_trans,trans);
		
		double rmse = 0;
		for (int pt=0; pt<a_trans.size(); pt++) {
			rmse += norm(a_trans[pt] - b_subset[pt]);
		}
		rmse = sqrt(rmse / (double)a_trans.size());
		
//		cout << "("<<_a_len<<","<<_a_off<<") -> ("<<_b_len<<","<<_b_off<<")   RMSE: " << rmse << endl;
		scores_to_matches[i].first = rmse;
	}
	sort(scores_to_matches.begin(), scores_to_matches.end());

	{
		//Show curvatures
		vector<double> a_sig,b_sig;
		a_sig = a_DB[scores_to_matches.front().second.queryIdx], 
		b_sig = b_DB[scores_to_matches.front().second.trainIdx];
		ShowMathGLCurves(a_sig, b_sig, "curvatures0");
	}
		
	a_len = a_DB_params[scores_to_matches.front().second.queryIdx].x;
	a_off = a_DB_params[scores_to_matches.front().second.queryIdx].y;
	b_len = b_DB_params[scores_to_matches.front().second.trainIdx].x;
	b_off = b_DB_params[scores_to_matches.front().second.trainIdx].y;
	score = scores_to_matches.front().first;
	
	cout << "("<<a_len<<","<<a_off<<") -> ("<<b_len<<","<<b_off<<")   RMSE: " << scores_to_matches.front().first << endl;
}


#pragma mark HSK extracting and matching

void CompareSignatures(const Mat_<double>& a, const Mat_<double>& b) {
	double max_cc = 0.0;
	int max_len,max_off,max_y,max_offb,max_yb;
	
	for (int len = 20; len < a.cols; len++) {
		//iterate different curve sizes, starting at 20 points
		cout << "len " << len <<  endl;
		
		for (int y = 0; y < a.rows; y++) {
			//iterate sigma values on A (rows of A signature)
			
			for (int yb = 0; yb<b.rows; yb++) {
				//iterate sigma values on B (rows of B signature)
				
				for (int off = 0; off < (a.cols - len); off++) {
					//iterate segments on A curve
					vector<double> templt; a(Range(y,y+1),Range(off,off+len)).copyTo(templt);
					assert(templt.size() == len);

					for (int offb = 0; offb < (b.cols - len); offb++) {
						vector<double> bv; b(Range(yb,yb+1),Range(offb,offb+len)).copyTo(bv);
						
						double cc = CalcCrossCorrelation(templt, bv) * (double)len;
						if (cc > max_cc) {
							max_cc = cc;
							max_y = y; max_off = off; max_len = len; max_offb = offb; max_yb = yb;
						}
					}
				}
			}
		}
	}
}

#define STRINGIFY(A) #A

#pragma mark Curvature Extrema Matching

Mat_<double> GetSmithWatermanHMatrix(const vector<pair<char,int> >& a, const vector<pair<char,int> >& b) {
	int M = a.size();
	int N = b.size();
	
	//Smith-Waterman
	Mat_<double> H(M+1,N+1,0.0);
	for (int i=1; i <= M; i++) {
		for (int j=1; j <= N; j++) {
			vector<double> v(4,0.0); 
			v[1] = H(i-1,j-1) + ((a[i-1].first == b[j-1].first) ? 2.0 : -1.0);
			v[2] = H(i-1,j) - 1.0;
			v[3] = H(i,j-1) - 1.0;
			H(i,j) = *(max_element(v.begin(), v.end()));
		}
	}
//	cout << H << endl;
	return H;
}	

/* original Smith Waterman algorithm */
double MatchSmithWaterman(const vector<pair<char,int> >& a, const vector<pair<char,int> >& b, vector<Point>& matching) 
{	
	vector<Point> traceback;
	Mat_<double> H = GetSmithWatermanHMatrix(a,b);
	Point maxp; double maxval;
	minMaxLoc(H, NULL, &maxval, NULL, &maxp);
	vector<char> step;
	while (H(maxp.y,maxp.x) != 0) {
		//				cout << "H(maxp.y-1,maxp.x-1) > H(maxp.y,maxp.x-1)" << H(maxp.y-1,maxp.x-1) << " > " << H(maxp.y,maxp.x-1) << endl;
		if (H(maxp.y-1,maxp.x-1) > H(maxp.y,maxp.x-1) &&
			H(maxp.y-1,maxp.x-1) > H(maxp.y-1,maxp.x)) 
		{
			traceback.push_back(maxp);
			maxp = maxp - Point(1,1);
			step.push_back('a');
		} else
			if (H(maxp.y-1,maxp.x) > H(maxp.y-1,maxp.x-1) &&
				H(maxp.y-1,maxp.x) > H(maxp.y,maxp.x-1)) 
			{
				traceback.push_back(maxp);
				maxp.y--;
				step.push_back('d');
			} else
				if (H(maxp.y,maxp.x-1) > H(maxp.y-1,maxp.x-1) &&
					H(maxp.y,maxp.x-1) > H(maxp.y-1,maxp.x)) 
				{
					traceback.push_back(maxp);
					maxp.x--;
					step.push_back('i');
				}
				else {
					//default - go back on both
					traceback.push_back(maxp);
					maxp = maxp - Point(1,1);
					step.push_back('a');
				}
	}
	for (vector<Point>::reverse_iterator it = traceback.rbegin(); 
		 it != traceback.rend() - 1; 
		 ++it) 
	{
		if((*it).y != (*(it+1)).y && (*it).x != (*(it+1)).x)
			matching.push_back(Point((*it).y,(*it).x));
	}
	for (vector<Point>::reverse_iterator it = traceback.rbegin(); 
		 it != traceback.rend(); 
		 ++it) 
	{
		if(it==traceback.rend())
			cout << a[(*it).y].first;
		else {
			if((*it).y == (*(it+1)).y)
				cout << "-";
			else {
				cout << a[(*it).y].first;
			}
		}
	} 
	cout << endl;
	for (vector<Point>::reverse_iterator it = traceback.rbegin(); 
		 it != traceback.rend(); 
		 ++it) 
	{
		if(it==traceback.rend())
			cout << b[(*it).x].first;
		else {
			if((*it).x == (*(it+1)).x)
				cout << "-";
			else
				cout << b[(*it).x].first;
		}
	} 
	cout << endl;
	for (int k=0; k<step.size(); k++) {
		cout << step[k];
	}
	cout << endl;
	return maxval;
}
