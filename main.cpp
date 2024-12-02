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
    cout << "共输入 " << picture << " 张图片" << endl;
    int flag = stitch();
    cout << "平均每张图片耗时 = " << time.toc() / picture / 1000 <<" 秒" << endl;
}