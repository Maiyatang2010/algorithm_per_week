#include "LDA.h"
#include "EigenvalueDecomposition.h"

// 构造函数
LDA::LDA(const string& feature_path, const int num_per_class, const int dim_size, const int num_class) :feature_path(feature_path), num_per_class(num_per_class), dim_size(dim_size), num_class(num_class) {}

// 行读取
vector<double> LDA::SplitString(const string& s, const string& c) {
	vector<double> v;
	string::size_type pos1, pos2;
	pos1 = 0;
	pos2 = s.find(c);
	while (string::npos != pos2) {
		v.push_back(stod(s.substr(pos1, pos2 - pos1)));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(stod(s.substr(pos1)));

	return v;
}

// 文件读取
vector<vector<double>> LDA::ReadFeatures(const string& feature_path)
{
	vector<vector<double>> feature_mat;

	string line;
	ifstream infile(feature_path);
	while (getline(infile, line)) {
		feature_mat.push_back(SplitString(line, ","));
	}

	return feature_mat;
}

//vec转mat
Mat LDA::Vec2Mat(vector<vector<double>>& feature_mat)
{
	int num_rows = feature_mat.size();
	int num_cols = feature_mat[0].size();
	Mat featMap(num_rows, num_cols, CV_64FC1);

	double *ptmp = NULL;
	for (int i = 0; i < num_rows; i++)
	{
		ptmp = featMap.ptr<double>(i);
		for (int j = 0; j < num_cols; j++)
		{
			ptmp[j] = feature_mat[i][j];
		}
	}

	return featMap;
}

// 训练
void LDA::LdaTrain()
{
	vector<vector<double>> feature_map = ReadFeatures(feature_path);
	Mat featMap = Vec2Mat(feature_map);

	Mat meanTotal = Mat::zeros(1, dim_size, CV_64FC1);
	vecotr<Mat> meanClass(num_class, Mat::zeros(1, dim_size, CV_64FC1));

	for (int i = 0; i < featMat.rows; i++) {
		Mat instance = featMat.row(i);
		int classIdx = i / num_per_class;
		add(meanTotal, instance, meanTotal);
		add(meanClass[classIdx], instance, meanClass[classIdx]);
	}

	// 计算整体均值
	meanTotal.convertTo(meanTotal, meanTotal.type(), 1.0 / static_cast<double> (featMat.rows));
	// 计算各类的均值
	for (int i = 0; i < num_class; i++)
	{
		meanClass[i].convertTo(meanClass[i], meanClass[i].type(), 1.0 / static_cast<double> (num_per_class));
	}

	// 计算Sw矩阵
	for (int i = 0; i < featMat.rows; i++)
	{
		int classIdx = i / num_per_class;
		Mat instance = featMat.row(i);
		subtract(instance, meanClass[classIdx], instance);
	}
	Mat Sw = Mat::zeros(dim_size, dim_size, featMat.type());
	mulTransposed(data, Sw, true);

	// 计算Sb矩阵
	Mat Sb = Mat::zeros(D, D, featMat.type());
	for (int i = 0; i < num_class; i++)
	{
		Mat tmp;
		subtract(meanClass[i], meanTotal, tmp);
		mulTransposed(tmp, tmp, true);
		add(Sb, tmp, Sb);
	}

	// Sw逆矩阵
	Mat Swi = Sw.inv();

	// M = inv(Sw) * Sb
	Mat M;
	gemm(Swi, Sb, 1.0, Mat(), 0.0, M);

	// 奇异值分解
	EigenvalueDecomposition es(M);
	eigenvalues = es.eigenvalues();
	eigenvectors = es.eigenvectors();

	eigenvalues = eigenvalues.reshape(1, 1);
	std::vector<int> sorted_indices = argsort(eigenvalues, false);

	eigenvalues = sortMatrixColumnsByIndices(eigenvalues, sorted_indices);
	eigenvectors = sortMatrixColumnsByIndices(eigenvectors, sorted_indices);

	eigenvalues = Mat(eigenvalues, Range::all(), Range(0, dim_size));
	eigenvectors = Mat(eigenvectors, Range::all(), Range(0, dim_size));
}