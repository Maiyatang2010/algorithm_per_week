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
	vector<double> SplitString(const string& s, const string& c);			// �з���תΪ������
	vector<vector<double>> ReadFeatures(const string& feature_path);		// �����������ļ�
	Mat Vec2Mat(vector<vector<double>>& feature_map);						// ��άvectorתMat
	void LdaTrain();														// ������������������ֵ

private:
	const string feature_path;				// ѵ������ַ
	const int num_per_class;				// ÿ��������
	const int num_class;					// �����
	const int dim_size;						// ����ά��
	Mat eigenvalues;						// ����ֵ
	Mat eigenvectors;						// ��������
};