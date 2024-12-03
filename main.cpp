#ifndef CONFIG_H
#define CONFIG_H
#include "include.h"
#endif

vector<string> img_names;
/*
如果通过摄像头视野+飞机高度+飞行速度
来计算跳帧数量，是最好的

一次10张：2.1s
一次12张：3.1s
一次14张：4s

特征点稀少的地方不能跳太多
*/

int picture = 1; // 组计数器

int main()
{
    int skepCount = 30; // 跳跃帧数

    // string data_path = "C:/Users/Administrator/Desktop/stitch_code/result/*.jpg";
    string data_path = "C:/Users/Administrator/Desktop/fly8/fly8/jpg1/*.jpg";

    vector<string> pre_path;
    glob(data_path, pre_path);
    // img_names.push_back(pre_path[0]);

    int count = 0; // 图片计数器
    while (true)
    {
        TicToc time;
        int skep = 8; // 每组skep张图
        cout << "第 " << picture << " 张图片" << endl;
        while (skep--)
        {
            img_names.push_back(pre_path[count]);
            count += skepCount;
        }
        int flag = stitch();
        
        count -= skepCount * 2;

        img_names.clear();
        picture++;
        cout << "平均每张图片耗时 = " << time.toc() / 5 / 1000 << " 秒" << endl << endl;
    }
}