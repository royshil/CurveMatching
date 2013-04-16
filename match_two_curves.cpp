using namespace cv;

#include "CurveCSS.h"
#include "CurveSignature.h"

int main(int argc, char** argv) {
	vector<Point> a,b;
	
	Mat src = imread("deer-18.png");
	if (src.empty()) {
		cerr << "can't read image" << endl; exit(0);
	}
	GetCurveForImage(src, a, false);
	ResampleCurve(a, a, 200, false);
	
	vector<Point2d> a_p2d;
	ConvertCurve(a, a_p2d);
	
	//create the target curve
	{
		//rotate and scale
		Scalar meanpt = mean(a);
		Mat_<double> trans_to = getRotationMatrix2D(Point2f(meanpt[0],meanpt[1]), 5, 0.65);
		
		//trasnlate
		trans_to(0,2) += 40;
		trans_to(1,2) += 40;
	
		vector<Point2d> b_p2d;
		cv::transform(a_p2d,b_p2d,trans_to);
		
		// everybody in the house - make some noise!
		cv::RNG rng(27628);
		for (int i=0; i<b_p2d.size(); i++) {
			b_p2d[i].x += (rng.uniform(0.0,1.0) - 0.5) * 20;
			b_p2d[i].y += (rng.uniform(0.0,1.0) - 0.5) * 20;
		}
		
		ConvertCurve(b_p2d, b);
		
		// occlude
		vector<Point> b_occ;
		for (int i=50; i<130; i++) {
			b_occ.push_back(b[i]);
		}
		ResampleCurve(b_occ, b, 200, true);
	}
	
	//Compare curves
	int a_len,a_off,b_len,b_off;
	double db_compare_score;
	CompareCurvesUsingSignatureDB(a, 
								  b,
								  a_len,
								  a_off,
								  b_len,
								  b_off,
								  db_compare_score
								  );

	//Get matched subsets of curves
	vector<Point> a_subset(a.begin() + a_off, a.begin() + a_off + a_len);
	vector<Point> b_subset(b.begin() + b_off, b.begin() + b_off + b_len);
	
	//Normalize to equal length
	ResampleCurve(a_subset, a_subset, 200, true);
	ResampleCurve(b_subset, b_subset, 200, true);
		
	//Visualize the original and target
	Mat outout(src.size(),CV_8UC3,Scalar::all(0));
	{
		//draw small original
		vector<Point2d> tmp_curve;
		cv::transform(a_p2d,tmp_curve,getRotationMatrix2D(Point2f(0,0),0,0.2));
		Mat tmp_curve_m(tmp_curve); tmp_curve_m += Scalar(25,0);
		drawOpenCurve(outout, tmp_curve, Scalar(255), 1);
		
		//draw small matched subset of original
		ConvertCurve(a_subset, tmp_curve);
		cv::transform(tmp_curve,tmp_curve,getRotationMatrix2D(Point2f(0,0),0,0.2));
		Mat tmp_curve_m1(tmp_curve); tmp_curve_m1 += Scalar(25,0);
		drawOpenCurve(outout, tmp_curve, Scalar(255,255), 2);

		//draw small target
		ConvertCurve(b, tmp_curve);
		cv::transform(tmp_curve,tmp_curve,getRotationMatrix2D(Point2f(0,0),0,0.2));
		Mat tmp_curve_m2(tmp_curve); tmp_curve_m2 += Scalar(outout.cols - 150,0);
		drawOpenCurve(outout, tmp_curve, Scalar(255,0,255), 1);

		//draw big target
		drawOpenCurve(outout, b, Scalar(0,0,255), 1);
		//draw big matched subset of target
		drawOpenCurve(outout, b_subset, Scalar(0,255,255), 1);
	}
	
	
	//Prepare the curves for finding the transformation
	vector<Point2f> seq_a_32f,seq_b_32f,seq_a_32f_,seq_b_32f_;

	ConvertCurve(a_subset, seq_a_32f_);
	ConvertCurve(b_subset, seq_b_32f_);
	
	assert(seq_a_32f_.size() == seq_b_32f_.size());
	
	seq_a_32f.clear(); seq_b_32f.clear();
	for (int i=0; i<seq_a_32f_.size(); i++) {
//		if(i%2 == 0) { // you can use only part of the points to find the transformation
			seq_a_32f.push_back(seq_a_32f_[i]);
			seq_b_32f.push_back(seq_b_32f_[i]);
//		}
	}
	assert(seq_a_32f.size() == seq_b_32f.size()); //just making sure
	
	vector<Point2d> seq_a_trans(a.size());
	
	//Find the fitting transformation
	//	Mat affineT = estimateRigidTransform(seq_a_32f,seq_b_32f,false); //may wanna use Affine here..
	Mat trans = Find2DRigidTransform(seq_a_32f, seq_b_32f);
	cout << trans;
	cv::transform(a_p2d,seq_a_trans,trans);
	
	//draw the result matching : the complete original curve as matched to the target 
	drawOpenCurve(outout, seq_a_trans, Scalar(0,255,0), 2);
	
	
	//May want to visualize point-by-point matching
//	cv::transform(seq_a_32f,seq_a_32f,trans);
//	for (int i=0; i<seq_a_32f.size(); i++) {
//		line(outout, seq_a_32f[i], seq_b_32f[i], Scalar(0,0,255), 1);
//	}
	
	imshow("outout", outout);
	
	waitKey();
	
}