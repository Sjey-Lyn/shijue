#include "rp_kalman.h"


// 构造函数
// 参数：状态值数目，测量值数目，控制系数，
// 注意：_q和_R的大小比例会影响预测结果，一般两者的比值不可以超过10^4，否则会过分相信矫正值而忽略了真实值
// 一般设置_q/_R在1~100之间
kalman_filter::kalman_filter(int _state_num, int _measure_num, int _control_param, double _q, double _R)
{
    q = _q;
    R = _R;
    state_num = _state_num;
    measure_num = _measure_num;
    control_param = _control_param;
    KF.init(state_num, measure_num, control_param);
}
// 获取单位矩阵
// 参数：单位矩阵的大小（整数N）
// 返回值：N*N的单位矩阵
Mat kalman_filter::get_diagonal_mat(int size)
{
    Mat E = Mat::zeros(size, 1, CV_32F);
    for(int i = 0; i < size; i++)
    {
        E.at<float>(i,i) = 1;
    }
    return E;
}
// 对一个值矫正的初始化函数接口
// 参数：输入矫正对象的初始值
void kalman_filter::init_kalman_filter(double first_value)
{
    KF.transitionMatrix = (get_diagonal_mat(state_num));  //转移矩阵A
    setIdentity(KF.measurementMatrix);                                             //测量矩阵H
    setIdentity(KF.processNoiseCov, Scalar::all(q));                            //系统噪声方差矩阵Q
    setIdentity(KF.measurementNoiseCov, Scalar::all(R));                        //测量噪声方差矩阵R
    setIdentity(KF.errorCovPost, Scalar::all(1));
    measurement = Mat::zeros(measure_num, 1, CV_32F);
    KF.statePost = (Mat_<float>(1, 1) << first_value);
}
// 对一个值的矫正函数
// 参数：输入矫正对象的值
// 返回值：矫正过后的值
double kalman_filter::correct_value(double value)
{
    KF.predict();
    measurement.at<float>(0) = value;
    KF.correct(measurement);
    return KF.statePost.at<float>(0);
}
// 对多个值矫正的初始化函数接口
// 参数：输入矫正对象的初始值（Mat数组）
void kalman_filter::init_kalman_filter(Mat first_value)
{
    KF.transitionMatrix = (get_diagonal_mat(state_num));  //转移矩阵A
    setIdentity(KF.measurementMatrix);                                             //测量矩阵H
    setIdentity(KF.processNoiseCov, Scalar::all(q));                            //系统噪声方差矩阵Q
    setIdentity(KF.measurementNoiseCov, Scalar::all(R));                        //测量噪声方差矩阵R
    setIdentity(KF.errorCovPost, Scalar::all(1));
    measurement = Mat::zeros(measure_num, 1, CV_32F);
    KF.statePost = first_value;
}
// 对多个值的矫正函数
// 参数：输入矫正对象的值（Mat数组）
// 返回值：矫正过后的值
Mat kalman_filter::correct_value(Mat value)
{
    KF.predict();
    measurement = value;
    KF.correct(measurement);
    return KF.statePost;
}

// 传输数据给电控过程以及打弹过程中延迟时间
#define DELAY_TIME 0.005
// 枪口的半径
#define CARHEAD_RADIUS 0.1
// 对世界坐标x、y、z的矫正系数
#define PREDICT_ALPHA 1
#define PREDICT_BETA 1
#define PREDICT_LAMDA 1
// 对敌方车辆的速度的矫正器
kalman_filter speed_kf;

// 预测函数
// 参数：前后帧信息类，以及采用的模式
// 返回值：Mat<double>数组，第1个值是angel_yx，第2个值是angel_zx。
Mat object_predict(frame_info &pre_frame, frame_info &now_frame, int mode)
{
    if(now_frame.flag == 0)
    {
        return (Mat_<double>(1, 2) << 0.0, 0.0);
    }
    double x,y,z;
    double sin_xy, cos_xy;
    double angel_yx, angel_zx;
    x = now_frame.tVec.at<double>(0,0);
    y = now_frame.tVec.at<double>(0,1);
    z = now_frame.tVec.at<double>(0,2);
    if(pre_frame.flag == 0)
    {
        angel_yx = fastAtan2(y,x);
        angel_zx = fastAtan2(z,x);
        return (Mat_<double>(1, 2) << angel_yx, angel_zx);
    }
    double move_dist = norm(pre_frame.tVec,now_frame.tVec,NORM_L2);
    double speed = move_dist/now_frame.delta_T;
    // 考虑小车沿x,y平面方向前行，并对预测结果进行校正
    if(mode==1)
    {
        speed = speed_kf.correct_value(speed);
        double len = sqrt(pow(x,2)+pow(y,2));
        sin_xy = y/len;
        cos_xy = x/len;
        x += speed*cos_xy*PREDICT_ALPHA*DELAY_TIME;
        y += speed*sin_xy*PREDICT_BETA*DELAY_TIME;
    }
    // 考虑小车沿x,y平面方向前行，且考虑了车头自身的旋转，并对预测结果进行校正
    else if(mode==2)
    {
        speed = PREDICT_ALPHA*speed -
                PREDICT_BETA*(now_frame.carhead_angel_speed*CARHEAD_RADIUS);
        speed = speed_kf.correct_value(speed);
        double len = sqrt(pow(x,2)+pow(y,2));
        sin_xy = y/len;
        cos_xy = x/len;
        x += speed*cos_xy*DELAY_TIME;
        y += speed*sin_xy*DELAY_TIME;
    }
    angel_yx = fastAtan2(y,x);
    angel_zx = fastAtan2(z,x);

//    cout<< x << "   " << y << "     " << z << endl;
    // angel_yx --> 上下
    // angel_zx --> 左右
    return (Mat_<double>(1, 2) << angel_yx, angel_zx);
}
