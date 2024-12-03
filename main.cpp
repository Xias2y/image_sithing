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
    TicToc time;
    int skepCount = 50; // 跳跃帧数

    string data_path = "C:/Users/Administrator/Desktop/stitch_code/data0/*.jpg";
    vector<string> pre_path;
    glob(data_path, pre_path);
    // img_names.push_back(pre_path[0]);

    int count = 0; // 图片计数器
    while (true)
    {
        int skep = 10; // 每组10张图
        cout << "第 " << picture << " 张图片" << endl;
        while (skep--)
        {
            img_names.push_back(pre_path[count]);
            count += skepCount;
        }
        int flag = stitch();
        
        img_names.clear();
        picture++;
    }


    // cout << "共输入 " << picture << " 张图片" << endl;
    int flag = stitch();
     cout << "平均每张图片耗时 = " << time.toc() / picture / 1000 <<" 秒" << endl;
}