#ifndef CONFIG_H
#define CONFIG_H
#include "include.h"
#endif

vector<string> img_names;
/*
���ͨ������ͷ��Ұ+�ɻ��߶�+�����ٶ�
��������֡����������õ�

һ��10�ţ�2.1s
һ��12�ţ�3.1s
һ��14�ţ�4s
*/

int main()
{
    TicToc time;
    int skepCount = 75;

    string data_path = "C:/Users/Administrator/Desktop/stitch_code/data1/*.jpg";
    vector<string> pre_path;
    glob(data_path, pre_path);
    img_names.push_back(pre_path[0]);
    
    int picture = 1;

    for (int count = skepCount; count <= pre_path.size(); count += skepCount)
    {
        img_names.push_back(pre_path[count]);
        picture++;
    }
    cout << "������ " << picture << " ��ͼƬ" << endl;
    int flag = stitch();
    cout << "ƽ��ÿ��ͼƬ��ʱ = " << time.toc() / picture / 1000 <<" ��" << endl;
}