/* 
���б�ǩ��������LDA��ά
���У�����������������ݰ���۴أ�ÿ��������num_per_class
*/

#include "LDA.h"
#include <iostream>

using namespace std;

int main()
{
	string file_path;			//�����ļ���ַ
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
