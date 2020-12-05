#include<iostream>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;
#ifndef RP_KALMAN_H
#define RP_KALMAN_H

// 帧信息类
struct frame_info{
    int flag;               // 识别状态
    RotatedRect armor_rect; // 装甲板旋转矩形
    Rect roi_rect;          //roi矩形
    Mat tVec;               //pnp得到的平移矩阵
    double carhead_angel_speed; //电控传输过来的车头角速度
    double delta_T;         //两帧时间差
};
Mat object_predict(frame_info &pre_frame, frame_info &now_frame, int mode=1);
// 卡尔曼滤波器类的定义
class kalman_filter
{
public:
    int state_num;
    int measure_num;
    int control_param;
    // 系统噪声方差
    double q;
    // 测量噪声方差
    double R;

    Mat measurement;
    KalmanFilter KF;

    kalman_filter(int _state_num = 1, int _measure_num = 1, int _control_param = 0, double _q = 1e-4, double _R = 1e-3);
    Mat get_diagonal_mat(int size);
    void init_kalman_filter(double first_value);
    double correct_value(double value);
    void init_kalman_filter(Mat first_value);
    Mat correct_value(Mat value);

};
#endif // RP_KALMAN_H
