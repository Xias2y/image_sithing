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

������ϡ�ٵĵط�������̫��
*/

int picture = 1; // �������

int main()
{
    int skepCount = 30; // ��Ծ֡��

    // string data_path = "C:/Users/Administrator/Desktop/stitch_code/result/*.jpg";
    string data_path = "C:/Users/Administrator/Desktop/fly8/fly8/jpg1/*.jpg";

    vector<string> pre_path;
    glob(data_path, pre_path);
    // img_names.push_back(pre_path[0]);

    int count = 0; // ͼƬ������
    while (true)
    {
        TicToc time;
        int skep = 8; // ÿ��skep��ͼ
        cout << "�� " << picture << " ��ͼƬ" << endl;
        while (skep--)
        {
            img_names.push_back(pre_path[count]);
            count += skepCount;
        }
        int flag = stitch();
        
        count -= skepCount * 2;

        img_names.clear();
        picture++;
        cout << "ƽ��ÿ��ͼƬ��ʱ = " << time.toc() / 5 / 1000 << " ��" << endl << endl;
    }
}