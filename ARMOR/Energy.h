#pragma once

#include <opencv2/opencv.hpp>
#include <rp_kalman.h>
using namespace cv;//这是个不好的编程习惯，尽量改用cv::

extern RotatedRect center_R;
extern const double PI;
extern int SCREEN_MAX_X;//屏幕分辨率1280x720
extern int SCREEN_MAX_Y;

class RMvector
{
	// 待优化
	// 这个应该是可以改成Point2f的派生类，这个想法有点欠妥，xy冲突
public:
	// 没有用到起点和终点的信息
	//Point2f start_point = Point2f(0, 0);// 起始点
	//Point2f end_point = Point2f(0, 0);	// 终止点
	//Point2f vec;						// 以原点为中心的向量
	double x;							// 以原点为中心的x坐标
	double y;							// 以原点为中心的y坐标
	double angle;						// 向量的角度，以横向为基准
	RMvector()
	{
		//start_point = Point2f(0, 0);
		//end_point = Point2f(0, 0);
		x = 0; y = 0;
		angle = 0;
	}
	RMvector(Point2f& start, Point2f& end)
	{
		//start_point = start;
		//end_point = end;
		//Point2f vec = end_point - start_point;
		Point2f vec = end - start;
		x = vec.x; y = vec.y;
		angle = atan(y / x);// 误差的可能来源
	}
	RMvector(double a)// 由角度计算单位向量
	{
		x = cos(a * PI / 180); y = sin(a * PI / 180);
		angle = a;
	}
	double getNorm()// 计算向量的长度
	{
		return sqrt(x * x + y * y);
	}
	double dot(RMvector& new_vector)// 向量点乘另外一个向量
	{
		return x * new_vector.x + y * new_vector.y;
	}
	double cross(RMvector& new_vector)// 向量叉乘另外一个向量，注意顺序
	{
		return x* new_vector.y - y * new_vector.x;
	}
};

//class RCenter: public RotatedRect // 继承旋转矩形
//{
//	// 这个继承貌似并没有什么卵用，算了，不继承了
//	// 圆心有什么特殊的函数吗
//	// 主要为了代码封装而写，主要为了代码的整洁性，现在暂不考虑
//};

class ArmorRect
{
	// 这个可以改成旋转矩形的派生类
public:
	Point2f center;					// 旋转矩形的中心
	double width;					// 旋转矩形的width
	double height;					// 旋转矩形的height
	double long_edge;				// 长边
	double short_edge;				// 短边
	double rotatedrect_angle;		// 旋转矩形原有的角度(-90, 0]
	double angle;					// 装甲板的角度，[0, 180);	//由这两个角度就可以计算切线和法线的向量了
	double tangent_angle;			// 装甲板切线角度
	double area;					// 面积
	RMvector to_center;				// 向心向量
	RMvector tangent;				// 切线
	double speed;					// 装甲板的角速度，每帧多少rad
	//RotatedRect rotRect;
	bool hitted;					// 该装甲板是否被点亮标志位 // 外部接口
	RMvector vertex2center[4];
	Point2f vertices[4];
	ArmorRect() {};
	ArmorRect(RotatedRect& rect)
	{
		//rotRect = rect;
		center = rect.center;
		width = rect.size.width;
		height = rect.size.height;
		long_edge = max(width, height);
		short_edge = min(width, height);
		rotatedrect_angle = rect.angle;
		area = width * height;
		setAngel();
		setTangent();
		//RMvector vec(center, center_R.center);// 这里可以重载()运算符来简化代码，一个小优化，不是必要
		//to_center = vec;
		speed = 0;
		hitted = 0;

		rect.points(vertices);

	}
	void setParam(RotatedRect& rect)// 用于更新新的装甲板
	{
		//rotRect = rect;
		center = rect.center;
		width = rect.size.width;
		height = rect.size.height;
		long_edge = max(width, height);
		short_edge = min(width, height);
		rotatedrect_angle = rect.angle;
		area = width * height;
		setAngel(); // 设置角度
		setTangent();
		//RMvector vec(center, center_R.center);// 这里可以重载()运算符来简化代码
		//to_center = vec;
		speed = 0;
		hitted = 0;

		rect.points(vertices);
	}
	void setSpeed(double w)
	{
		speed = w;
	}
	void setAngel()
	{
		// 已稳
		angle = width > height ? (rotatedrect_angle + 90) :  rotatedrect_angle;
		//std::cout << "angle: " << angle << std::endl;
	}
	void setTangent()
	{
		// 已稳
		tangent_angle = width < height ? (rotatedrect_angle + 90) : rotatedrect_angle;
		//std::cout << "tangent: " << tangent_angle << std::endl;
	}
	void setToCenter()
	{		
		RMvector vec(center, center_R.center);// 这里可以重载()运算符来简化代码，一个小优化，不是必要

		to_center = vec;

		for (int i = 0; i < 4; i++)
		{
			RMvector vec(vertices[i], center_R.center);
			vertex2center[i] = vec; // 可以直接赋值，值传递
		}
	}
};

typedef struct Roi
{
	Point center;
	int width;
	int height;
	Point Anchor;		// ROI 直立矩形的rect的xy
	double scale = 1.5;		// 暂时为1，保留原状
	void getAnchor()
	{
		Anchor.x = (int)center.x - width / 2;
		Anchor.y = (int)center.y - height / 2;
	}
	Roi()// 无参构造函数 // 用于构造整个屏幕的ROI
	{
		center = Point(SCREEN_MAX_X / 2, SCREEN_MAX_Y / 2);
		width = SCREEN_MAX_X; height = SCREEN_MAX_Y;
		Anchor = Point(0, 0);
	}
	// 根据旋转装甲版构造ROI
	Roi(ArmorRect& rect)
	{
		center = rect.center;
		width = scale * (int)rect.long_edge;
		height = scale * (int)rect.long_edge;
		getAnchor();

		//防止越界
		if (Anchor.x < 0)
			Anchor.x = 0;
		if (Anchor.y < 0)
			Anchor.y = 0;
		if (Anchor.x + width > SCREEN_MAX_X) {
			width = SCREEN_MAX_X - Anchor.x;
		}
		if (Anchor.y + height > SCREEN_MAX_Y) {
			height = SCREEN_MAX_Y - Anchor.y;
		}
	}

	Roi(std::vector<Point> &contour)
	{
		Rect rect = boundingRect(contour);
		center = Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
		width = scale * (int)rect.width;
		height = scale * (int)rect.height;
		getAnchor(); // 由于根据中心缩放，所以这一步是必要的

		//防止越界
		if (Anchor.x < 0)
			Anchor.x = 0;
		if (Anchor.y < 0)
			Anchor.y = 0;
		if (Anchor.x + width > SCREEN_MAX_X) {
			width = SCREEN_MAX_X - Anchor.x;
		}
		if (Anchor.y + height > SCREEN_MAX_Y) {
			height = SCREEN_MAX_Y - Anchor.y;
		}
	}
	void drawRoi(Mat& frame)// 在原图画出roi
	{
		rectangle(frame, Rect(Anchor.x, Anchor.y, width, height), Scalar(0, 255, 255), 2);
	}
}Roi;

// 此类主要是对能量机关输入的两个装甲板类来做处理
// 一个是当前的装甲板，一个是前一个装甲板
class Energy
{
private:
	int THRESH_BR = 56;
	int THRESH_GRAY = 79;
	int MIN_AREA = 200;						// 所有可识别物的最小接受面积
	int IS_RED = 0;							// 敌方装甲板颜色为红色则为1
	int SHUN = 0;							// 判断是顺逆时针，0检测不出，1为顺，2为逆
	bool BIG_OR_SMALL = 0;					// 小符，1为大符
	double NEWSPEED;						// 装甲板的新速度
	kalman_filter kf;						// 定义卡尔曼滤波器
	double accelerate;						// 当前加速度(由前后两帧的速度计算得到)
	double former_speed;					// 前一帧的速度
	double second_to_frame = 0.03;			// 一帧多少秒
	double bias = 0;						// 扰动的大小
	int DILATE_EOFF = 7;
	Roi roi;

	// 上次调车的时候发现会由掉帧的问题，可以设置ROI跟踪圆心
	// 先实现加速度大符扰动预测
public:
	// 需要返回给电控的值
	float pre_x = 0;						// 预测击打的x坐标
	float pre_y = 0;						// 预测击打的y坐标
	int is_find = 0;						// 是否找到装甲板
	// 偏移时间
	float getTime;
	Energy() 
	{
		kf.init_kalman_filter(3);				// 卡尔曼滤波速度初始化
	}
	// 需要一些拷贝构造函数
	// 通过每一帧图像得到一个mask。图像识别主要函数
	void videoProcess(Mat& frame, Mat& mask, ArmorRect& present, ArmorRect& former)
	{
		getMask(frame, mask);					// 得到掩膜
		/* ------从mask中找出轮廓确定present------- */
		getCoutour(mask, frame, present);		// 找装甲板的轮廓 // 得到新的present
		getR(mask, present);					// 得到中心R	// 圆心需要装甲板来确定 // 确定装甲版的向心向量
		// 只有完成前面两步操作才可以确定present
		present.setToCenter();
		//if (is_find)
		//	roi.drawRoi(frame);
		if (is_find)
		{
			judgeDirect(present, former, frame);	// 判断顺逆 
			newSpeedMeasure(present, former);
			getSpeed(present, former, frame);		// 计算角速度
			former = present;						// 保留装甲版信息
		}
		//predict(present, frame);				// 通过角速度进行预测
	}
	void reset()
	{
		//std::cout << pre_y << std::endl;//debug用
		pre_x = 0;
		pre_y = 0;
	}
	// 外部接口:设置阈值
	void setThresh(int br, int gray) {
		THRESH_BR = br;
		THRESH_GRAY = gray;
	}
	// 外部接口:是否为红色
	void isRed(bool b)
	{
		IS_RED = b;
	}
	
	void getMask(Mat& frame, Mat& mask);
	// 借助当前帧的装甲板和之前帧的装甲板计算角速度或者是角度，主要用到点乘
	void getSpeed(ArmorRect& present, ArmorRect& former, Mat& frame);
	// 预测需要击打的位置，需要返回pitch和yaw
	void predict(ArmorRect& present, Mat& frame);

	// 判断是顺时针还是逆时针
	void judgeDirect(ArmorRect& present, ArmorRect& former, Mat& frame);
	// 通过轮廓查找的方法找到装甲板的位置
	void getCoutour(Mat& mask, Mat& frame, ArmorRect& armor_center);
	// 这个还可以调调，识别R中心时的一系列判断
	bool isValidCenterRContour(const std::vector<Point>& center_R_contour);
	// 得到R中心的函数，得到center_R的坐标
	void getR(Mat& mask, ArmorRect& present);
	bool judgeRPosition(RotatedRect& R, ArmorRect& present);


	// 把数据封装起来发送给电控
	void send(double &x, double &y)
	{
		if (is_find)
		{
			x = pre_x;
			y = pre_y;
		}
		else
		{
			x = 0;
			y = 0;
		}
	}
	bool isValidCenterFansContour(const std::vector<Point>& countour);
	bool isValidCenterArmorContour(const std::vector<Point>& contour);
	void activeBig()
	{
		BIG_OR_SMALL = 1;
	}
	void activeSmall()
	{
		BIG_OR_SMALL = 0;
	}
	void newSpeedMeasure(ArmorRect& present, ArmorRect& former)
	{
		// 这个算法需要计算半径，半径也很重要
		double h = fabs(present.center.y - center_R.center.y);
		double w = fabs(present.center.x - center_R.center.x);
		//double r = sqrt(h * h + w * w);
		double delta_h = fabs(present.center.y - former.center.y);
		double delta_w = fabs(present.center.x - former.center.x);
		double radio1 = delta_h / w / PI * 180;
		double radio2 = delta_w / h / PI * 180;
		if (radio1 < 10 && radio2 < 10)
		{
			if (h > 2 * w)
				NEWSPEED = radio2;
			else if (w > 2 * h)
				NEWSPEED = radio1;
			else
				NEWSPEED = (delta_w / h + delta_h / w) / 2 / PI * 180;
		}
	}
	void bigPredict(ArmorRect& present, ArmorRect& former, Mat& frame)
	{
		if (!is_find) return;
		int frameNum = 10;// 3帧延迟
		// 旋转的角度，做的是向量的旋转
		RMvector vec = present.to_center;
		//double theta = frameNum * 3 * PI / 180;// present.speed;
		//std::cout << "speed: " << present.speed << std::endl;
		double res = cummulativeAngel(frameNum, former.speed, present.speed);
		double theta = res;
		//double theta = (frameNum * 1.305 * second_to_frame);
		//double theta = (cummulativeAngel(frameNum, former.speed, present.speed) + (frameNum * 1.305 / second_to_frame)) * PI / 180;
		//std::cout << "res: " << res << std::endl;
		std::cout << "theta: " << theta << std::endl;
		double newx, newy;
		if (SHUN == 1)
		{
			theta = -theta;
			newx = center_R.center.x - (vec.x * cos(theta) + vec.y * sin(theta));
			newy = center_R.center.y - (-vec.x * sin(theta) + vec.y * cos(theta));
			circle(frame, Point(newx, newy), 2, Scalar(0, 255, 0), 2);
			pre_x = newx; pre_y = newy;
		}
		else if (SHUN == 2)
		{
			newx = center_R.center.x - (vec.x * cos(theta) + vec.y * sin(theta));
			newy = center_R.center.y - (-vec.x * sin(theta) + vec.y * cos(theta));
			circle(frame, Point(newx, newy), 2, Scalar(0, 255, 0), 2);
			pre_x = newx; pre_y = newy;
		}
	}
	void bigPredict2(ArmorRect& present, ArmorRect& former, Mat& frame)
	{
		if (!is_find) return;
		int frameNum = 10;// 3帧延迟
		// 旋转的角度，做的是向量的旋转
		RMvector vec = present.to_center;
		//double theta = frameNum * 3 * PI / 180;// present.speed;
		//std::cout << "speed: " << present.speed << std::endl;
		double second = frameNum * second_to_frame;
		double res = bigIntAngel(present, former, second);
		double theta = res;
		std::cout << "res:: " << res << std::endl;
		double newx, newy;
		if (SHUN == 1)
		{
			theta = -theta;
			newx = center_R.center.x - (vec.x * cos(theta) + vec.y * sin(theta));
			newy = center_R.center.y - (-vec.x * sin(theta) + vec.y * cos(theta));
			circle(frame, Point(newx, newy), 2, Scalar(0, 255, 0), 2);
			pre_x = newx; pre_y = newy;
		}
		else if (SHUN == 2)
		{
			newx = center_R.center.x - (vec.x * cos(theta) + vec.y * sin(theta));
			newy = center_R.center.y - (-vec.x * sin(theta) + vec.y * cos(theta));
			circle(frame, Point(newx, newy), 2, Scalar(0, 255, 0), 2);
			pre_x = newx; pre_y = newy;
		}
	}

	void updateSpeed(ArmorRect& present, ArmorRect& former, Mat& frame)
	{
		
		// y = 0.785*sin(1.884*t)+1.305
		double Speed = present.speed;// 如果没有找到装甲板，则这个速度将是之间的装甲板的速度(已更新的速度)
		if (Speed > 0.1)//设为0.1真的好吗
		{
			Speed = kf.correct_value(Speed);
			//energy.setSpeed(Speed);
			present.setSpeed(Speed); // 更新速度
			predict(present, frame);				// 小符预测
			//bigPredict(present, former, frame); // 大符预测 --- 近似方案
			//bigPredict2(present, former, frame); // 大符预测 --- 函数方案
			putText(frame, "UPDATE: " + std::to_string(Speed), Point(100, 100), CV_FONT_NORMAL, 1, Scalar(0, 255, 0), 2);

		}

	}
	void calZL(ArmorRect& present)
	{
		// 计算帧率
	}

	// 待用(大符：双线程 + 硬触发)
	void getAccelerate(ArmorRect& present, ArmorRect& former)
	{
		double acc = (present.speed - former.speed);// 计算加速度
		double omega = 1.884 * second_to_frame;// 转化单位
		bias = acc / (omega * omega);//计算扰动，扰动应该是当前帧转动的角度，也可以作为下一帧的角度

	}
	double cummulativeAngel(int num, double f_speed, double speed)// 迭代num次
	{
		// 弃用，算法不正确
		double sum_angel = 0;
		double omega = 1.884;
		double b;
		double pre_speed = speed;
		double angel;
		double acc;
		for (int i = 0; i < num; i++)
		{	
			//angel = (pre_speed + f_speed)/2; // 用中位数近似
			acc = (pre_speed - f_speed) * second_to_frame; // 时间为1帧
			b = acc / (omega * omega);
			std::cout << "b: " << b << std::endl;
			// 迭代下一帧
			f_speed = pre_speed;
			sum_angel += b;
			//b = sum_angel + b;
			pre_speed += b * omega * omega / second_to_frame;

		}
		return sum_angel;
	}

	double bigIntAngel(ArmorRect& present, ArmorRect& former, double second)
	{
		/*------------------------------------第一套方案-----------------------------------------*/
		//double acc = (present.speed - former.speed) * PI / 180 / second_to_frame;// 计算加速度
		////std::cout << "speed1: " << present.speed << " formerSpeed: " << former.speed << std::endl;
		////double omega = 1.884 ;// 转化单位
		////double b = acc / (omega * omega);//计算扰动，扰动应该是当前帧转动的角度，也可以作为下一帧的角度
		//double t_0 = acos(-acc  / 0.785) / 1.884;
		//double b = -0.785 / 1.884 * cos(1.884 * t_0);
		//double t = t_0 + second;
		//double b_pre = -0.785 / 1.884 * cos(1.884 * t);
		//return b_pre - b;
		/*------------------------------------第二套方案-----------------------------------------*/
		//double t_0 = asin(present.speed  * PI / 180 / 0.785) / 1.884 / second_to_frame;
		////double t_0 = acos(-b * 1.884 / 0.785) / 1.884;
		//double b = -0.785 / 1.884 * cos(1.884 * t_0);
		//double t = t_0 + second;
		//double b_pre = -0.785 / 1.884 * cos(1.884 * t);
		//return b_pre - b;
	}
};
