CurveMatching
=============

Matching 2D curves in OpenCV


See http://www.morethantechnical.com/2012/12/26/2d-curve-matching-w-code/ for details.

Sample Usage
------------

	vector<Point> a,b;

	//Get curve from image
	GetCurveForImage("binary_image_with_silhuette.png", a, false);
	ResampleCurve(a, a, 200, false);
	
	vector<Point2d> a_p2d;
	ConvertCurve(a, a_p2d);
	
	//rotate and scale
	Scalar meanpt = mean(a);
	Mat_<double> trans_to = getRotationMatrix2D(Point2f(meanpt[0],meanpt[1]), 5, 0.65);

	//create target curve from original curve
	vector<Point2d> b_p2d;
	cv::transform(a_p2d,b_p2d,trans_to);
	
	ConvertCurve(b_p2d, b);


	//Compare curves
	int a_len,a_off,b_len,b_off;
	double compare_score;
	CompareCurvesUsingSignatureDB(a, 
				  b,
				  a_len,
				  a_off,
				  b_len,
				  b_off,
				  compare_score
				  );

	vector<Point2d> a_subset(a.begin() + _a_off, a.begin() + _a_off + _a_len);
	vector<Point2d> b_subset(b.begin() + _b_off, b.begin() + _b_off + _b_len);
	
	ResampleCurve(a_subset, a_subset, 200, true);
	ResampleCurve(b_subset, b_subset, 200, true);
	
	Mat trans = Find2DRigidTransform(a_subset, b_subset);
	cout << trans << endl;
	vector<Point2d> a_trans;
	cv::transform(a_subset,a_trans,trans);



Compile
-------
	mkdir build
	cd build
	cmake ..
	make

