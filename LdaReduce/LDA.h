#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv.hpp>
#include <core\core.hpp>

using namespace cv;
using namespace std;


class LDA
{
public:
	LDA(const string& feature_path, const int num_per_class, const int dim_size, const int num_class);
	vector<double> SplitString(const string& s, const string& c);			// 切分行转为行向量
	vector<vector<double>> ReadFeatures(const string& feature_path);		// 读特征向量文件
	Mat Vec2Mat(vector<vector<double>>& feature_map);						// 二维vector转Mat
	void LdaTrain();														// 计算特征向量和特征值

private:
	const string feature_path;				// 训练集地址
	const int num_per_class;				// 每类样本数
	const int num_class;					// 类别数
	const int dim_size;						// 样本维度
	Mat eigenvalues;						// 特征值
	Mat eigenvectors;						// 特征向量
};