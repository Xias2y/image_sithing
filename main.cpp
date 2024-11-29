#ifndef CONFIG_H
#define CONFIG_H
#include "include.h"
#endif

vector<string> img_names;

int main()
{
    TicToc time;
    int skepCount = 5;

    string data_path = "C:/Users/Administrator/Desktop/stitch_code/data1/*.jpg";
    vector<string> pre_path;
    glob(data_path, pre_path);
    img_names.push_back(pre_path[0]);
    
    int picture = 1;

    for (int count = skepCount; count <= pre_path.size(); count += skepCount)
    {
        img_names.push_back(pre_path[count]);
        cout << "第 " << picture << " 张图片：" << endl;
        picture++;
        int flag = stitch();
        if (flag != 0) break;
    }

    cout << "平均每张图片耗时 = " << time.toc() / picture / 1000 <<" 秒" << endl;
}