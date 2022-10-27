// https://blog.csdn.net/cfan927/article/details/108884457
cv::Mat Interate(
	cv::Mat image,
	cv::Mat xs,
	cv::Mat ys,
	double alpha,
	double beta,
	double gamma,
	double kappa,
	double wl,
	double we,
	double wt,
	int iterations)
{
	// 相关参数
	int N = iterations;
	cv::Mat smth = image.clone();

	// 图像大小
	qDebug() << "Calculating size of image";
	cv::Size size = image.size();
	int row = size.height;
	int col = size.width;

	// 计算外部力（图像力）
	qDebug() << "Computing external forces";
	cv::Mat E_line = smth.clone(); // E_line is simply the image intensities

	cv::Mat gradx, grady;
	cv::Sobel(smth, gradx, smth.depth(), 1, 0, 1, 1, 0, cv::BORDER_CONSTANT);
	cv::Sobel(smth, grady, smth.depth(), 0, 1, 1, 1, 0, cv::BORDER_CONSTANT);

	qDebug() << "Computing gradx and grady";
	cv::Mat E_edge(row, col, CV_32FC1);
	for (int i = 0; i < gradx.rows; i++)
	{
		for (int j = 0; j < gradx.cols; j++)
		{
			float v_gradx = gradx.at<float>(i, j);
			float v_grady = grady.at<float>(i, j);
			
			E_edge.at<float>(i, j) = -1 * std::sqrt(v_gradx * v_gradx + v_grady * v_grady); // E_edge is measured by gradient in the image
		}
	}

	// 导数mask
	qDebug() << "masks for taking various derivatives";
	cv::Mat m1 = (cv::Mat_<float>(1, 2) << -1, 1);
	cv::Mat m2 = (cv::Mat_<float>(2, 1) << -1, 1);
	cv::Mat m3 = (cv::Mat_<float>(1, 3) << 1, -2, 1);
	cv::Mat m4 = (cv::Mat_<float>(3, 1) << 1, -2, 1);
	cv::Mat m5 = (cv::Mat_<float>(2, 2) << 1, -1, -1, 1);

	cv::Mat cx, cy, cxx, cyy, cxy;
	filter2D(smth, cx, -1, m1);
	filter2D(smth, cy, -1, m2);
	filter2D(smth, cxx, -1, m3);
	filter2D(smth, cyy, -1, m4);
	filter2D(smth, cxy, -1, m5);

	// 计算 E_term
	cv::Mat E_term(row, col, CV_32FC1);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			int v_cx = cx.at<float>(i, j);
			int v_cy = cy.at<float>(i, j);
			int v_cxx = cxx.at<float>(i, j);
			int v_cyy = cyy.at<float>(i, j);
			int v_cxy = cxy.at<float>(i, j);

			E_term.at<float>(i, j) = (v_cyy*v_cx*v_cx - 2 * v_cxy*v_cx*v_cy + v_cxx * v_cy*v_cy) / (std::pow((1 + v_cx * v_cx + v_cy * v_cy), 1.5));
		}
	}

	// 计算E_ext
	cv::Mat E_ext = (wl*E_line + we * E_edge - wt * E_term);

	// 计算梯度
	cv::Mat fx, fy;
	cv::Sobel(E_ext, fx, E_ext.depth(), 1, 0, 1, 0.5, 0, cv::BORDER_CONSTANT);
	cv::Sobel(E_ext, fy, E_ext.depth(), 0, 1, 1, 0.5, 0, cv::BORDER_CONSTANT);

	cv::transpose(xs, xs);
	cv::transpose(ys, ys);

	int m = xs.rows;
	int n = 1;

	int mm = fx.cols;
	int nn = fx.rows;

	// 计算五对角状矩阵，b(i)表示vi系数(i = i - 2 到 i + 2)
	double b[5];
	b[0] = beta;
	b[1] = -(alpha + 4 * beta);
	b[2] = 2 * alpha + 6 * beta;
	b[3] = b[1];
	b[4] = b[0];

	cv::Mat A = cv::Mat::eye(m, m, CV_32FC1);
	cv::Mat eyeMat0 = cv::Mat::eye(m, m, CV_32FC1);
	circRowShift(eyeMat0, 2);
	eyeMat0.convertTo(eyeMat0, CV_32FC1);
	A = b[0] * eyeMat0;

	cv::Mat eyeMat1 = cv::Mat::eye(m, m, CV_32FC1);
	circRowShift(eyeMat1, 1);
	eyeMat1.convertTo(eyeMat1, CV_32FC1);
	A = A + b[1] * eyeMat1;

	cv::Mat eyeMat2 = cv::Mat::eye(m, m, CV_32FC1);
	circRowShift(eyeMat2, 0);
	eyeMat2.convertTo(eyeMat2, CV_32FC1);
	A = A + b[2] * eyeMat2;

	cv::Mat eyeMat3 = cv::Mat::eye(m, m, CV_32FC1);
	circRowShift(eyeMat3, -1);
	eyeMat3.convertTo(eyeMat3, CV_32FC1);
	A = A + b[3] * eyeMat3;

	cv::Mat eyeMat4 = cv::Mat::eye(m, m, CV_32FC1);
	circRowShift(eyeMat4, -2);
	eyeMat4.convertTo(eyeMat4, CV_32FC1);
	A = A + b[4] * eyeMat4;

	// 计算矩阵的逆
	cv::Mat Ainv(A.size(), CV_32FC1);
	A = A + gamma * cv::Mat::eye(m, m, CV_32FC1);
	cv::invert(A, Ainv); //  Computing Ainv 

	cv::Mat srcImg = cv::imread("D:/endo_image.jpg");
	// 迭代更新曲线
	for (int i = 0; i < N; i++)
	{
		cv::Mat intFx(fx.size(), CV_32FC1);
		cv::Mat intFy(fy.size(), CV_32FC1);

		cv::remap(fx, intFx, xs, ys, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
		cv::remap(fy, intFy, xs, ys, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

		cv::Mat ssx(xs.size(), CV_32FC1);
		cv::Mat ssy(ys.size(), CV_32FC1);
		for (int k = 0; k < xs.rows; k++)
		{
			for (int l = 0; l < xs.cols; l++)
			{
				ssx.at<float>(k, l) = gamma * xs.at<float>(k, l) - kappa * intFx.at<float>(k, l);
				ssy.at<float>(k, l) = gamma * ys.at<float>(k, l) - kappa * intFy.at<float>(k, l);
			}
		}

		// 更新曲线位置
		xs = Ainv * ssx;
		ys = Ainv * ssy;

		cv::Mat resultImg = srcImg.clone();
		for (int j = 0; j < xs.rows; j++)
		{
			cv::Point center = cv::Point(xs.at<float>(j, 0), ys.at<float>(j, 0));
			cv::circle(resultImg, center, 4, cv::Scalar(0, 255, 0), 2);
		}

		// 显示
		cv::imshow("result", resultImg);
		cv::waitKey(30);
	}

	return image;
}
