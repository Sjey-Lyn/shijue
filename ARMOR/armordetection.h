#ifndef ARMORDETECTION_H
#define ARMORDETECTION_H

#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>
#include<queue>
#include<math.h>
#include<algorithm>

#define frame_H 720  // 摄像头获取的图片的高
#define frame_W 1280  // 摄像头获取的图片的宽
#define armor_h  62.5    // 3号装甲板的实际高（单位毫米）
#define armor_w  117.75    // 3号装甲板的实际宽（单位毫米）

using namespace std;
using namespace cv;

class Light
{
private:
    vector<Point> _contour;

public:
    Light(vector<Point>& contour);

    //获取灯条面积
    int get_area();
    //获取旋转矩形
    RotatedRect get_rotatderect();
    //获取灯条的中心点(注意是在roi下的坐标，要想获得原图坐标，还要加上roi的top—left点)
    Point get_center();
    //获取灯条最左点(注意是在roi下的坐标，要想获得原图坐标，还要加上roi的top—left点)
    Point Left_point();
    //获取灯条最右点(注意是在roi下的坐标，要想获得原图坐标，还要加上roi的top—left点)
    Point Right_point();
    //获取灯条最上点(注意是在roi下的坐标，要想获得原图坐标，还要加上roi的top—left点)
    Point Top_point();
    //获取灯条最下点(注意是在roi下的坐标，要想获得原图坐标，还要加上roi的top—left点)
    Point Bottom_point();
    // 将灯条拟合成直线，并返回该直线的斜率
    double line_k();
};


class Armordetection
{
public:
    Mat _img, _frame, _roi, _mask;
    int _flag = 0;      // 如果没有检测到装甲板则为0
    Rect _now_roi;      // 当前帧要处理的roi (上一帧传给当前帧的roi）
    Rect _next_roi;     // 当前帧要传给下一帧的roi
    RotatedRect _rotaterect;  // 当前帧识别到的打击目标的旋转矩形，如果没有识别到目标就为空
    vector<vector<Point>> _target_two_light; // 存放打击目标的两个灯条
    int _lossFrameK;      // 掉帧系数
    double _dt;         // //两帧时间差
    double _fps;       // 记录帧率
    Mat _tVec;            // pnp求解的平移矩阵
    Mat _num_img;
    double distance = 0;

    double yaw_angle = 0;
    double pitch_angle = 0;

public:
    Armordetection(Mat img, Rect roi, int loss_frame_k);
    void set_img(bool use_roi);
    void color_detection(int color_detection_mode, bool show_mask);
    //筛选灯条
    vector<vector<Point>> light_filter(bool show_light_filter_result);
    void show_img();
    // 匹配灯条
    vector<vector<Point>> light_match(vector<vector<Point>>& contours_filter, bool show_match_result);
    //在找到的所有装甲板中选择最优的打击目标
    //要确保Match.size>=2才能使用这个函数
    vector<Point> choose_target(vector<vector<Point>>& Match);
    Rect get_roi(vector<Point>& armor_4point, double roi_scale);
    // 获取平移矩阵
    Mat get_tVec(vector<Point> armor_4point, Mat c,  Mat d, double h, double w);
    Mat get_num_img(int img_size, bool show_num_img);
    void begin_to_detect(Mat c, Mat d, double roi_scale, double armorH, double armorW,
                         bool use_roi, int color_detection_mode, bool show_mask,
                         bool show_light_filter_result, bool show_match_result,
                         bool get_num_image, int num_img_size, bool show_num_img);

};



#endif // ARMORDETECTION_H
