/* 
对有标签的样本做LDA降维
其中，假设输入的样本数据按类聚簇，每类样本数num_per_class
*/

#include "LDA.h"
#include <iostream>

using namespace std;

int main()
{
	string file_path;			//特征文件地址
	int num_per_class, dim_size, num_class;
	
	cout << "Enter the path for feature file:" << endl;
	getline(cin, file_path);

	cout << "Enter values of: num_per_class, dim_size, num_class:" << endl;
	cin >> num_per_class >> dim_size >> num_per_class;
	
	cout << "initialize LDA ..." << endl;
	LDA lda(file_path, num_per_class, dim_size, num_class);
	
	cout << "LDA training" << endl;
	lda.LdaTrain();

	cin.get();
}
