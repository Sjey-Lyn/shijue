#include<iostream>
#include<opencv2/opencv.hpp>
#include"GxIAPI.h"
#include "DxImageProc.h"
#include<ImageConsProd.h>
#include<armordetection.h>
#include<rp_kalman.h>
#include<queue>
#include<serialport.h>
#include<CRC_Check.h>
#include<Energy.h>
using namespace cv;
using namespace std;


int mode = 1, buff_four, shoot_speed, my_color;

//相机参数的设定
GX_DEV_HANDLE hDevice = NULL;   //设备句柄
#define EXPOSURE  10000
#define GAIN 0
#define WHITEBALANCE 1
//#define GAIN_SET              //增益
//#define WHITEBALANCE_SET

#define VIDEO_WIDTH  1280       //相机分辨率
#define VIDEO_HEIGHT 720
#define V_OFFSET 304



volatile unsigned int prdIdx;
volatile unsigned int csmIdx;

#define BUFFER_SIZE 1

struct ImageData {
    Mat img;
    unsigned int frame;
};
ImageData data[BUFFER_SIZE];

VideoWriter vw("VideoTest.avi",VideoWriter::fourcc('M','P','4','2'),25,Size(1280,720),true); // 定义写入视频对象


SerialPort port("/dev/ttyUSB0");
VisionData vdata;




//回调函数
static void GX_STDC OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM* pFrame)
{
    int64_t nWidth;
    int64_t nHeight;

    nWidth = pFrame->nWidth;
    nHeight = pFrame->nHeight;

    nWidth = 1280;
    nHeight = 720;

    uchar* m_pBufferRaw = new uchar[nWidth * nHeight * 3]; ///< 原始图像数据（内存空间）
    uchar* pRGB24Buf = new uchar[nWidth * nHeight * 3];    ///< RGB图像数据（内存空间）

    if (pFrame->status == 0)                               // 正常帧，残帧返回-1
    {
        Mat src;
        src.create(VIDEO_HEIGHT, VIDEO_WIDTH, CV_8UC3);

        memcpy(m_pBufferRaw, pFrame->pImgBuf, pFrame->nImgSize);
        DxRaw8toRGB24(m_pBufferRaw, pRGB24Buf, (VxUint32)(pFrame->nWidth), (VxUint32)(pFrame->nHeight), RAW2RGB_NEIGHBOUR, DX_PIXEL_COLOR_FILTER(1), false);
        memcpy(src.data, pRGB24Buf, nHeight * nWidth * 3);
        cvtColor(src, src, CV_RGB2BGR);

        while (prdIdx - csmIdx >= BUFFER_SIZE)  // 不能注释，注释了会出现零图，然后导致掉帧发生
        {

        };

        data[prdIdx % BUFFER_SIZE].img = src;
        vw << src;
//        imshow("src",src);
        GXFlushQueue(hDevice);  //
        prdIdx++;
//        cout<<prdIdx<<endl;

    }

    delete[] m_pBufferRaw;
    delete[] pRGB24Buf;
    return;

}

int assert_success(GX_STATUS status)
{
    if (status != GX_STATUS_SUCCESS)
    {
        status = GXCloseDevice(hDevice);
        if (hDevice != NULL)
        {

        }
        status = GXCloseLib();
        return 0;
    }
}


void ImageConsProd::ImageProducer()
{
    // 主相机初始化
    GX_STATUS status = GX_STATUS_SUCCESS;
    GX_OPEN_PARAM stOpenParam;
    uint32_t nDeviceNum = 0;

    cout << "主相机初始化中......" << endl;

    status = GXCloseLib();
    status = GXInitLib();


    if (status != GX_STATUS_SUCCESS)
    {
        cout << "初始化失败!" << endl;
        cout << "错误码：" << status << endl;
        return ;
    }
    cout << "主相机初始化完成!" << endl;

    status = GXUpdateDeviceList(&nDeviceNum, 1000);
    if ((status != GX_STATUS_SUCCESS) || (nDeviceNum <= 0))
    {
        cout << "获取设备列表失败" << endl;
        return ;
    }
    cout << "设备数：" << nDeviceNum << endl;

    stOpenParam.accessMode = GX_ACCESS_EXCLUSIVE;
    stOpenParam.openMode = GX_OPEN_INDEX;
    stOpenParam.pszContent = "1";
    status = GXOpenDevice(&stOpenParam, &hDevice);

    if (status == GX_STATUS_SUCCESS)
    {
        //设置采集模式为连续采集
        status = GXSetEnum(hDevice, GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);
        assert_success(status);

        double val = 4000;
        status = GXGetFloat(hDevice, GX_FLOAT_EXPOSURE_TIME, &val);

        ////DUBUG_CHANGE_EXPOSURE
        cout << "默认曝光值：" << val << endl;
        val = EXPOSURE;
        cout << "val = " << val << endl;
        status = GXSetFloat(hDevice, GX_FLOAT_EXPOSURE_TIME, val);
        if (status == GX_STATUS_SUCCESS)
        {
            cout << "设置后曝光值：" << val << endl;
            assert_success(status);
        }
        else
            cout << "SET_EXPOSURE_FAIL" << endl;

        status = GXSetInt(hDevice, GX_INT_HEIGHT, VIDEO_HEIGHT);
        status = GXSetInt(hDevice, GX_INT_WIDTH, VIDEO_WIDTH);
        status = GXSetInt(hDevice, GX_INT_OFFSET_X, 100);
        status = GXSetInt(hDevice, GX_INT_OFFSET_Y, 304);
        assert_success(status);


#ifdef GAIN_SET
        //GAIN_SET
        status = GXSetEnum(hDevice, GX_ENUM_GAIN_SELECTOR, GX_GAIN_SELECTOR_ALL);
        if (status == GX_STATUS_SUCCESS){
            cout << "set_gain_succeed " << endl;
            assert_success(status);
        }else{
            cout << "增益_SELECT_fail:  " << status  << endl;
        }
        status = GXSetFloat(hDevice, GX_FLOAT_GAIN, GAIN);
        if (status == GX_STATUS_SUCCESS){
            cout << "set_gain_succeed_ggg " << endl;
            assert_success(status);
        }else{
            cout <<"增益 fail:  "  << status <<endl;
        }
#endif

#ifdef WHITEBALANCE_SET

        status = GXSetEnum(hDevice, GX_ENUM_BALANCE_RATIO_SELECTOR, GX_BALANCE_RATIO_SELECTOR_GREEN);
        if (status == GX_STATUS_SUCCESS){
            cout <<"set_whitebalance_succeed " <<endl;
            assert_success(status);
        }else{
            cout <<"whitebalance_SELECT_fail:  " << status  <<endl;
        }
        status = GXSetFloat(hDevice, GX_FLOAT_BALANCE_RATIO, WHITEBALANCE);
        if (status == GX_STATUS_SUCCESS){
            cout <<"set_whitebalance_succeed_ggg " <<endl;
            assert_success(status);
        }else{
            cout <<"whitebalance fail:  "  << status <<endl;
        }

#endif


        //设置触发开关为OFF
        status = GXSetEnum(hDevice, GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_OFF);
        assert_success(status);

        //注册图像处理回调函数

        status = GXRegisterCaptureCallback(hDevice, NULL, OnFrameCallbackFun);
        cout << "回调函数:" << status << endl;
        //发送开采命令
        GXSendCommand(hDevice, GX_COMMAND_ACQUISITION_START);
        while(1);

        //发送停采命令
        status = GXSendCommand(hDevice, GX_COMMAND_ACQUISITION_STOP);
        //注销采集回调
        status = GXUnregisterCaptureCallback(hDevice);
    }

    //在结束的时候调用GXCLoseLib()释放资源
    status = GXCloseDevice(hDevice);
    status = GXCloseLib();
}

void ImageConsProd::ImageConsumer()
{
    port.initSerialPort();
    Energy energy;
    energy.isRed(false);

    Mat frame;
    queue<Rect> last_roi;   //存放前几帧的roi
    queue<frame_info> frame_information;   //卡尔曼预处需要的信息
    last_roi.push(Rect(0,0, frame_W, frame_H));  // 第一帧roi是整张图片
    frame_info init_frame_information;
    init_frame_information.flag = 0;
    frame_information.push(init_frame_information); // 初始化一下，第一帧的前一帧没有目标，flag为0
    double roi_scale=0.4;   // roi相对于旋转矩形的放大比例
    int loss_frame = 0;      // 掉帧系数
    Mat kalman_result;       // 卡尔曼预测结果

    // 相机内参
    Mat cam = (Mat_<double>(3, 3) << 1849.8, 3.6044, 653.4334,
                                     0.0000000000000000, 1852.1, 485.3128,
                                     0.0000000000000000, 0.0000000000000000, 1.0000000000000000);
    // 畸变系数
    Mat dis = (Mat_<double>(5, 1) <<-0.0257, 0.6377, 0.0028,
                                    0.0034, 0.0000000000000000);

    while (1){

        while(prdIdx - csmIdx == 0);                                         //线程锁
        port.get_Mode(mode,buff_four,shoot_speed,my_color);
        data[csmIdx % BUFFER_SIZE].img.copyTo(frame);                         //将Producer线程生产的图赋值给src
        ++csmIdx;                                                           //解锁，生产下一张图
//        cout<<csmIdx<<endl;
        if(frame.empty()){
            continue;
        }
        imshow("frame", frame);
        if(mode==1)
        {
            Armordetection armordetection = Armordetection(frame, last_roi.back(), loss_frame);

            // 开始检测装甲板
    //        void begin_to_detect(Mat c, Mat d, double roi_scale, double armorH, double armorW,
    //                             bool use_roi, int color_detection_mode, bool show_mask,
    //                             bool show_light_filter_result, bool show_match_result,
    //                             bool get_num_image, int num_img_size, bool show_num_img)
            //Mat c, Mat d ------>相机内参, 畸变系数
            //double roi_scale -----> roi相对于旋转矩形的放大比例
            //double armorH, double armorW -----> 检测的装甲板的实际高，宽
            //bool use_roi -----> 是否使用roi，建议调试时不使用，以观察算法的准确性，比赛时使用，以提高检测速度
            //int color_detection_mode 图像预处理提取二值图的模式，目前只有模式1，后期再补充
            //bool show_mask ------> 是否展示图像预处理效果
            //bool show_light_filter_result, bool show_match_result ----> 是否展示筛选灯条，匹配灯条的效果
            // bool get_num_image -----> 是否提取装甲板上的数字, 如果否，则num_img_size, show_num_img随便给
            // int num_img_size, bool show_num_img ---> 提取装甲板上的数字的图像大小， 以及是否展示提取的数字

            armordetection.begin_to_detect(cam, dis, roi_scale, armor_h, armor_w, 1, 1, 1, 1, 1 ,1, 100, 1);
    //////////// 展示检测效果
            armordetection.show_img();

            vdata.dis.f = armordetection.distance;

            vdata.pitch_angle.f = -armordetection.pitch_angle*180/3.14;
            vdata.yaw_angle.f = -armordetection.yaw_angle*180/3.14;
            vdata.isFindTarget =armordetection._flag ;
            vdata.buff_change_four=0;
            vdata.anti_top=0;
            vdata.anti_top_change_armor = 0;
            vdata.nearFace = 0;
            vdata.isfindDafu= 0;

          }

           if(mode==2)
           {
               double val = 2000;
               cout << "val = " << val << endl;
               GX_STATUS status = GX_STATUS_SUCCESS;
               status = GXSetFloat(hDevice, GX_FLOAT_EXPOSURE_TIME, val);

               int THRESH_BR = 57;//56
               int THRESH_GRAY = 128;//128
               Mat mask;
               ArmorRect present, former;
               energy.getTime = 40;
               energy.setThresh(THRESH_BR, THRESH_GRAY);
               energy.videoProcess(frame, mask, present, former);
               //energy.pre_x = present.center.x;
               //energy.pre_y = present.center.y;
               energy.updateSpeed(present, former, frame);
               //energy.predict(present, src);

               vdata.pitch_angle.f = energy.pre_y;
               vdata.yaw_angle.f = energy.pre_x;
               vdata.isfindDafu =energy.is_find;

               namedWindow("Dafu");
               namedWindow("DafuBinary");
               imshow("Dafu", frame);
               imshow("DafuBinary", mask);

           }


        port.TransformData(vdata);
        port.send();



         waitKey(1);



//        //这里记录前10帧的roi信息
//        if ( last_roi.size() >= 10)
//        {
//            last_roi.pop();
//        }
//        last_roi.push(armordetection._next_roi);

//        // 计算掉帧系数
//        if(armordetection._flag == 1)
//        {
//            loss_frame = 0;
//        }
//        else
//        {
//            loss_frame++;
//        }

//        frame_info now_frame_information;
//        now_frame_information.flag =  armordetection._flag;
//        now_frame_information.armor_rect =  armordetection._rotaterect;
//        now_frame_information.roi_rect =  armordetection._now_roi;
//        now_frame_information.tVec =  armordetection._tVec;
//        now_frame_information.delta_T =  armordetection._dt;
////        now_frame_information.carhead_angel_speed =     ;

//        kalman_result = object_predict(frame_information.back(), now_frame_information, 1);
//        kalman_result.at<double>(0); // 上下
//        kalman_result.at<double>(1); // 左右


//        //这里记录前2帧的frame信息
//        if ( frame_information.size() >= 2)
//        {
//            frame_information.pop();
//        }
//        frame_information.push(now_frame_information);







    }
}


