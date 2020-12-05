#include<armordetection.h>


Light::Light(vector<Point>& contour)
{
    _contour = contour;
}

//获取灯条面积
int Light::get_area()
{
    return contourArea(_contour);
}

//获取旋转矩形
RotatedRect Light::get_rotatderect()
{
    return minAreaRect(_contour);
}

//获取灯条的中心点(注意是在roi下的坐标，要想获得原图坐标，还要加上roi的top—left点)
Point Light::get_center()
{
    Moments mu;
    mu = moments(_contour);
    return Point(mu.m10/mu.m00, mu.m01/mu.m00);
}

//获取灯条最左点(注意是在roi下的坐标，要想获得原图坐标，还要加上roi的top—left点)
Point Light::Left_point()
{
    Point Left = *min_element(this->_contour.begin(), this->_contour.end(),
      [](const Point& lhs, const Point& rhs) {return lhs.x < rhs.x;});
    return Left;
}

//获取灯条最右点(注意是在roi下的坐标，要想获得原图坐标，还要加上roi的top—left点)
Point Light::Right_point()
{
    Point Right = *max_element(this->_contour.begin(), this->_contour.end(),
      [](const Point& lhs, const Point& rhs) {return lhs.x < rhs.x;});
    return Right;
}

//获取灯条最上点(注意是在roi下的坐标，要想获得原图坐标，还要加上roi的top—left点)
Point Light::Top_point()
{
    Point Top = *min_element(this->_contour.begin(), this->_contour.end(),
      [](const Point& lhs, const Point& rhs) {return lhs.y < rhs.y;});
    return Top;
}

//获取灯条最下点(注意是在roi下的坐标，要想获得原图坐标，还要加上roi的top—left点)
Point Light::Bottom_point()
{    Point Bottom = *max_element(this->_contour.begin(), this->_contour.end(),
      [](const Point& lhs, const Point& rhs) {return lhs.y < rhs.y;});
     return Bottom;
}

// 将灯条拟合成直线，并返回该直线的斜率
double Light::line_k()
{
    Vec4f line_para;
    fitLine(_contour, line_para, DIST_L2, 0, 1e-2, 1e-2);
    double k = line_para[1] / line_para[0];
    return k;
}





Armordetection::Armordetection(Mat img, Rect roi, int loss_frame_k)
{
    _img = img.clone();
    _frame = img;
    _now_roi = roi;
    _lossFrameK = loss_frame_k;
}

void Armordetection::set_img(bool use_roi)
{
    if(use_roi){
        _roi = Mat(_img, _now_roi);
     }
     else
     {
        _roi = _img.clone();
        _now_roi = Rect(0, 0, _img.cols, _img.rows);
        _lossFrameK = 0;
     }
}

void Armordetection::color_detection(int color_detection_mode, bool show_mask)  // 后期再加入其他方法
{
//        // HSV
//        Mat temp, temp1,temp2, red, thres_whole;
//        vector<cv::Mat> channels;
//        cvtColor(_img, temp, COLOR_BGR2HSV);
//        split(temp, channels);
//        inRange(temp, Scalar(0,43,46),Scalar(11,255,255),temp1);
//        inRange(temp,Scalar(156,43,46),Scalar(181,255,255),temp2);
//        red = temp2 | temp1;
//        cvtColor(_img,thres_whole,CV_BGR2GRAY);
//        if (1)
//            threshold(thres_whole,thres_whole,33,255,THRESH_BINARY);
//        else
//            threshold(thres_whole,thres_whole,60,255,THRESH_BINARY);

//        imshow("thresh_whole", thres_whole);

//        Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3));
        Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3));

//        dilate(red, red, element1);
//        imshow("red", red);
//        Mat _max_color = red & thres_whole;  // _max_color获得了清晰的二值图
////        _max_color = rrr & thres_whole;  // _max_color获得了清晰的二值图
//        dilate(_max_color, _max_color, element2);
//        imshow("_max_color", _max_color);


   if (color_detection_mode == 1){
        inRange(_roi, Scalar(0, 0, 0), Scalar(170, 255, 255), _mask);
        _mask = ~_mask;
        Mat kernel1 = getStructuringElement(MORPH_RECT, Size(3, 3));
        Mat kernel2 = getStructuringElement(MORPH_RECT, Size(3, 3));
        dilate(_mask,_mask,element2);
//        morphologyEx(_mask, _mask, MORPH_CLOSE, kernel1);
//        morphologyEx(_mask, _mask, MORPH_OPEN, kernel2);
    }

   if(show_mask)
   {
       namedWindow("mask", WINDOW_NORMAL);
       imshow("mask", _mask);
   }
}


//筛选灯条
vector<vector<Point>> Armordetection::light_filter(bool show_light_filter_result)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hireachy;
    findContours(_mask, contours, hireachy, CV_RETR_TREE, CHAIN_APPROX_SIMPLE);

    vector<vector<Point>> contours_filter;
    RotatedRect rect;
    for(const vector<Point>& contour : contours)
    {
        rect=minAreaRect(contour);
        // k是旋转矩形的长宽比
        double k = min(rect.size.width, rect.size.height) / max(rect.size.width, rect.size.height);
        // 灯条的面积
        float lightContourArea = contourArea(contour);

        // 灯条面积太小的不要，注意这里可以再分细一点
        // 由于灯条刚刚转过来的时候，面积很小，w/h比也很小，可以结合这两个来筛选
        // 从而保证刚刚转过来的面积很小的灯条不会被筛选掉
        // 面积很小， w/h也要很小才保留
        if(lightContourArea < 280 && k > 0.4) {  /*cout<<"1"<<endl;*/   continue;}
        // 面积更小， w/h也要更小才保留
        if(lightContourArea < 47 && k > 0.3)  { /*cout<<"2"<<endl; */ continue;}

        // 灯条面积太大的不要
        if(lightContourArea > 5000) {/*cout<<"3"<<endl;*/ continue;}

        // 旋转矩形长宽比太大的灯条不要
        if(k>0.45)  {/*cout<<"4"<<endl;*/ continue;}

        // 灯条左右倾斜超过一定角度的不要
        if (rect.size.width == min(rect.size.width, rect.size.height) &&
            fabs(rect.angle) > 35 ) {/*cout<<"5"<<endl;*/continue;}
        if (rect.size.width == max(rect.size.width, rect.size.height) &&
            fabs(rect.angle) < 55 ) {/*cout<<"6"<<endl;*/ continue;}

        //灯条的凸度，即灯条面积比旋转矩形面积，对预处理要求较高，暂时不采用
        //if(lightContourArea /  rect.size.area() < 0.4) {/*cout<<"7"<<endl;*/ continue;}

        contours_filter.push_back(contour);
    }

    // 查看筛选后的灯条的结果
    if(show_light_filter_result)
    {
        if(contours_filter.size()>0){
        Mat output = Mat::zeros(_roi.size(), _roi.type());
        drawContours(output, contours_filter, -1, Scalar(255, 255, 255), -1);
        namedWindow("contours_filter", WINDOW_NORMAL);
        imshow("contours_filter", output);
        }
    }

    return contours_filter; //返回合格的灯条
}

void Armordetection::show_img()
{
    namedWindow("_img", WINDOW_NORMAL);
    imshow("_img", _img);
}


// 匹配灯条
vector<vector<Point>> Armordetection::light_match(vector<vector<Point>>& contours_filter, bool show_match_result)
{
    vector<vector<Point>> Match;  // 存放匹配好的灯条（总是两个一起连续存放）
    vector<int> Match_index;  //存放Match中灯条的索引（总是两个一起连续存放）
    int size = contours_filter.size();

    //灯条数大于2匹配才有意义，如果灯条数小于2，直接返回空的Match
    if (size >= 2){

    //按灯条中心x从大到小排序
    sort(contours_filter.begin(), contours_filter.end(), [](vector<Point>& light1,vector<Point>& light2)
          {  //Lambda函数,作为sort的cmp函数
        return Light(light1).get_center().x > Light(light2).get_center().x;});

    int temp[size][size] = {0};

    for(size_t i = 0; i < contours_filter.size(); i++)
    {

       Light light1 = Light(contours_filter.at(i));
       RotatedRect rect1=light1.get_rotatderect();
       // 灯条最上最下两个点
       Point top1 = light1.Top_point();
       Point bottom1 = light1.Bottom_point();
       double h1 = max(rect1.size.width, rect1.size.height); //灯条的长
       double w1 = min(rect1.size.width, rect1.size.height); //灯条的宽

       for(size_t j = 0; j < contours_filter.size(); j++)
       {
           if(i<=j) break; // 避免重复计算

           Light light2 = Light(contours_filter.at(j));
           RotatedRect rect2=light2.get_rotatderect();
           // 灯条最上最下两个点
           Point top2 = light2.Top_point();
           Point bottom2 = light2.Bottom_point();
           double h2 = max(rect2.size.width, rect2.size.height); //灯条的长
           double w2 = min(rect2.size.width, rect2.size.height); //灯条的宽

           //调节匹配条件的参数时，可以把一些条件注释掉，把单个匹配条件调到最优，再将所有匹配条件结合起来，增强鲁棒性
//                cout<<" "<<endl;
           //由于灯条按x坐标由大到小排序，且i>j,所以light1是左灯条，light2是右灯条
           // 两个灯条是“ \ \ ” 时，旋转矩形的角度差不能太大
           if (rect1.size.width == w1 &&
               rect2.size.width == w2 &&
               fabs(rect1.angle - rect2.angle) < 10 )
           {  /*cout<<"1"<<endl;  cout<<"1: "<<rect1.angle<<" "<<rect2.angle<<endl;*/
               temp[i][j] = temp[i][j] + 1; }
           // 两个灯条是“ / / ” 时，旋转矩形的角度差不能太大
           else if (rect1.size.width == h1 &&
                    rect2.size.width == h2 &&
                    fabs(rect1.angle - rect2.angle) < 10 )
           {  /*cout<<"2"<<endl; cout<<"2: "<<rect1.angle<<" "<<rect2.angle<<endl;*/
               temp[i][j] = temp[i][j] + 1; }
           // 两个灯条是“ \ / ” 时，旋转矩形的角度差不能太大
           else if (rect1.size.width == w1 &&
                    rect2.size.width == h2 &&
                    90 - fabs(rect1.angle - rect2.angle) < 8)
           {  /*cout<<"3"<<endl; cout<<"3: "<<rect1.angle<<" "<<rect2.angle<<endl;*/
               temp[i][j] = temp[i][j] + 1; }
           // 两个灯条是“ / \ ” 时，旋转矩形的角度差不能太大
           else if (rect1.size.width == h1 &&
                    rect2.size.width == w2 &&
                    90 - fabs(rect1.angle - rect2.angle) < 8 )
           {  /*cout<<"4"<<endl;  cout<<"4: "<<rect1.angle<<" "<<rect2.angle<<endl;*/
                temp[i][j] = temp[i][j] + 1; }

           // 两个灯条的长度要差不多
           if ((double)min(h1, h2)/(double)max(h1, h2) > 0.79)
           { /*cout<<"5"<<endl;  cout<<"5: "<<(double)min(h1, h2)/(double)max(h1, h2)<<endl;*/
             temp[i][j] = temp[i][j] + 1; }

           // 两个灯条的中心点y坐标要差不多（这里只是粗略计算）
           if (fabs (top1.y+bottom1.y - top2.y-bottom2.y) <  2 * (double)(h1+h2)/(4))
           { /*cout<<"6"<<endl;
             cout<<"6: "<<fabs (top1.y+bottom1.y - top2.y-bottom2.y)<<"<"<<2*(double)(h1+h2)/(4)<<endl;*/
             temp[i][j] = temp[i][j] + 1; }

           // 两个灯条组成的旋转矩形长宽比要合理（这里只是粗略计算）
           double w = max(abs(top1.x-top2.x), abs(bottom1.x-bottom2.x));
           double h = max(abs(top1.y-bottom1.y), abs(top2.y-bottom2.y));
           if(w/h < 4.5 && w/h > 1)
           {  /*cout<<"7"<<endl;*/  temp[i][j] = temp[i][j] + 1; }


           //两个灯条的顶点之间，底点之间连线的斜率不能太大
           double k1, k2;
           if ( top1.x-top2.x != 0)
           {
              k1 = fabs((double)(top1.y-top2.y)/(double)(top1.x-top2.x));
           }
           else {k1 = 10;} //用k1=10来表示斜率无穷大
           if ( bottom1.x-bottom2.x != 0)
           {
              k2 = fabs((double)(bottom1.y-bottom2.y)/ (double)(bottom1.x-bottom2.x));
           }
           else {k2 = 10;} //用k1=10来表示斜率无穷大
           if(k1 < 0.56 && k2 < 0.56)
           { /*cout<<"8"<<endl;*/  temp[i][j] = temp[i][j] + 1; }
       }
    }

    // 上面投票结果储存在temp中
       for(size_t i = 0; i < contours_filter.size(); i++)
       {
           for(size_t j = 0; j < contours_filter.size(); j++)
           {
               if(i<=j) break; // 避免重复计算

               //如果temp=5说明这索引为i，j的两个灯条已经满足了配对条件
               if(temp[i][j] == 5)
               {

                   //查找这两个灯条是否已经在Match向量当中
                   int c1 = count(Match.begin(), Match.end(), contours_filter[i]);
                   int c2 = count(Match.begin(), Match.end(), contours_filter[j]);

                   //如果两个都不在Match当中
//                       c1 = 0; c2 = 0;     //令c1,c2=0可以查看单靠以上八个条件的匹配效果
                   if(c1 == 0 && c2 ==0)
                   {
                       //把这两个灯条放进Match，并把他们的索引号i，j放进Match_index
                       Match.push_back(contours_filter.at(i));
                       Match.push_back(contours_filter.at(j));
                       Match_index.push_back(i);
                       Match_index.push_back(j);
                   }


                  // 如果两个灯条至少有一个已经被放进Match，说明有一个灯条同时与两个灯条配对了
                  // 解决思路是找出这两对灯条，再匹配一次，提出更严格的条件，最后只保留最优的那一对
                  else
                   {
                        for(size_t m = 0; m < Match_index.size(); m++)
                        {
                            int index1, index2, m1, m2;
                            vector<int>::iterator it;
                            vector<vector<Point>>::iterator it2;

                            // 如果第i或j个灯条之前被放进Match过
                            if (i == Match_index.at(m) || j == Match_index.at(m))
                            {
                               // index1，index2是 第i(j)个灯条 和第i(j)个灯条匹配的灯条 的索引
                               // 因为匹配的灯条总是一对一对放进Match的，所以要看m的奇偶
                               if(m%2 == 0)
                               {m1 = m; m2 = m+1; index1 = Match_index.at(m); index2 = Match_index.at(m+1);}
                               else {m1 = m-1; m2 = m; index1 = Match_index.at(m-1); index2 = Match_index.at(m);}

                               //现在要看的是i，j这一对灯条和index1，index2这对灯条哪个更加匹配，就保留哪一对

                               // 现在的条件是哪对灯条中心点的欧氏距离更小哪对就最优
                               // 单用这个条件效果就已经很好了
                               // 后期如果需要可以再加入灯条角度等条件
                               Light light_i = Light(contours_filter.at(i));
                               Light light_j = Light(contours_filter.at(j));
                               double distance_ij = abs(light_i.get_center().x - light_j.get_center().x)+
                                                    abs(light_i.get_center().y - light_j.get_center().y);

                               Light light_index1 = Light(contours_filter.at(index1));
                               Light light_index2 = Light(contours_filter.at(index2));
                               double distance_index = abs(light_index1.get_center().x - light_index2.get_center().x) +
                                                       abs(light_index1.get_center().y - light_index2.get_center().y);

                               if(distance_index > distance_ij)
                               {

                                   it=Match_index.begin();
                                   Match_index.erase(it+m1);
                                   Match_index.erase(it+m1);
                                   it2=Match.begin();
                                   Match.erase(it2+m1);
                                   Match.erase(it2+m1);

                                   Match.push_back(contours_filter.at(i));
                                   Match.push_back(contours_filter.at(j));
                                   Match_index.push_back(i);
                                   Match_index.push_back(j);
                               }
                               break;
                            }
                        }
                   }

               }
           }
       }
     }

    if (show_match_result){
        if(Match.size()>=2){
        Mat output=Mat::zeros(_roi.size(), _roi.type());
        drawContours(output, Match, -1, Scalar(255, 255, 255), -1);
        namedWindow("contours_match", WINDOW_NORMAL);
        imshow("contours_match", output);
        }
    }

    return Match;
}


//在找到的所有装甲板中选择最优的打击目标
//要确保Match.size>=2才能使用这个函数
vector<Point> Armordetection::choose_target(vector<vector<Point>>& Match)
{
    vector<int> s;
    vector<RotatedRect> rr;
    vector<vector<Point>> armor_4point;
    vector<vector<Point>> target_two_light;
    int ii, ind=0;

    for(size_t i = 0; i<Match.size(); i=i+2)
    {
        Light light1 = Light(Match.at(i));
        Light light2 = Light(Match.at(i+1));
        Point top1 = light1.Top_point() + _now_roi.tl();
        Point bottom1 = light1.Bottom_point() + _now_roi.tl();
        Point top2 = light2.Top_point() + _now_roi.tl();
        Point bottom2 = light2.Bottom_point() + _now_roi.tl();

        vector<Point> vPolygonPoint;
        vPolygonPoint.push_back(top1);
        vPolygonPoint.push_back(bottom1);
        vPolygonPoint.push_back(bottom2);
        vPolygonPoint.push_back(top2);

        //画旋转矩形
        RotatedRect armor_rect = minAreaRect(vPolygonPoint);
        Point2f vertex[4];
        armor_rect.points(vertex);
        for (int j = 0; j < 4; j++)
        {
            line(_img, vertex[j], vertex[(j + 1) % 4], Scalar(0, 0, 255), 2, CV_AA);
        }

        s.push_back(armor_rect.size.width * armor_rect.size.height); //保存旋转矩形面积
        rr.push_back(armor_rect);          // 保存旋转矩形
        armor_4point.push_back(vPolygonPoint);   // 保存一对灯条的四个顶点
        target_two_light.push_back(Match.at(i));      // 保存一对灯条
        target_two_light.push_back(Match.at(i+1));
     }

    //找出面积最大的旋转矩形的索引
    for(ii=1; ii<s.size(); ii++)
    {
        if(s[ii]>s[ind])
        {
            ind = ii;
        }
    }
   circle(_img, rr.at(ind).center, 8, Scalar(0,0,255), -1);
   _target_two_light.push_back(target_two_light.at(2*ind));
   _target_two_light.push_back(target_two_light.at(2*ind+1));

   return armor_4point.at(ind);
}

Rect Armordetection::get_roi(vector<Point>& armor_4point, double roi_scale)
{

   int xmax =0, xmin = 10000, ymax = 0, ymin = 10000, rw=0, rh=0;  //初始化一下

    Point top11 = armor_4point.at(0);
    Point bottom11 = armor_4point.at(1);
    Point top22 = armor_4point.at(3);
    Point bottom22 = armor_4point.at(2);

    int xmin1 = min(top11.x, top22.x);
    int xmin2 = min(bottom11.x, bottom22.x);
    xmin = min(xmin1, xmin2);

    int ymin1 = min(top11.y, top22.y);
    int ymin2 = min(bottom11.y, bottom22.y);
    ymin = min(ymin1, ymin2);

    int xmax1 = max(top11.x, top22.x);
    int xmax2 = max(bottom11.x, bottom22.x);
    xmax = max(xmax1, xmax2);

    int ymax1 = max(top11.y, top22.y);
    int ymax2 = max(bottom11.y, bottom22.y);
     ymax = max(ymax1, ymax2);

    xmin = max(1, (int)(xmin - 2*(xmax-xmin)*roi_scale));
    ymin = max(1, (int)(ymin - 2*(ymax-ymin)*roi_scale));

    rw = min((int)(xmax - xmin +(xmax-xmin)*roi_scale), _img.cols-xmin);
    rw = max(1, rw);
    rh = min((int)(ymax - ymin +(ymax-ymin)*roi_scale), _img.rows-ymin);
    rh = max(1, rh);

    Rect roi_rect = Rect(xmin, ymin, rw, rh);

    return roi_rect;
}

// 获取平移矩阵
Mat Armordetection::get_tVec(vector<Point> armor_4point, Mat c,  Mat d, double h, double w)
{
   Point top11 = armor_4point.at(0);
   Point bottom11 = armor_4point.at(1);
   Point top22 = armor_4point.at(3);
   Point bottom22 = armor_4point.at(2);

   circle(_img, top11, 4, Scalar(0,0,255), -1);
   putText(_img, "top1", top11, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
   circle(_img,  bottom11, 4, Scalar(0,0,255), -1);
   putText(_img, "bottom1", bottom11, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
   circle(_img, top22, 4, Scalar(0,0,255), -1);
   putText(_img, "top2", top22, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
   circle(_img, bottom22, 4, Scalar(0,0,255), -1);
   putText(_img, "bottom2", bottom22, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));

   //自定义的物体世界坐标，单位为mm
   vector<Point3f> obj=vector<Point3f>
   {
   Point3f(-w, -h, 0),	//tl
   Point3f(w, -h, 0),	//tr
   Point3f(w, h, 0),	//br
   Point3f(-w, h, 0)	//bl
   };
   vector<Point2f> pnts=vector<Point2f>
   {
   Point2f(top11),	    //tl
   Point2f(top22),	    //tr
   Point2f(bottom22),	//br
   Point2f(bottom11)	//bl
   };

   Mat rVec = Mat::zeros(3, 1, CV_64FC1); //init rvec
   Mat tVec = Mat::zeros(3, 1, CV_64FC1); //init tvec
   //进行位置解算
   solvePnP(obj,pnts,c,d,rVec,tVec,false, SOLVEPNP_ITERATIVE);
   distance = sqrt(tVec.at<double>(0)*tVec.at<double>(0)+tVec.at<double>(1)*tVec.at<double>(1)+
                   tVec.at<double>(2)*tVec.at<double>(2));

   yaw_angle = atan(tVec.at<double>(0)/tVec.at<double>(2));
   pitch_angle = atan(tVec.at<double>(1)/tVec.at<double>(2));
   string str_d = (string)("distances: ")+ to_string(distance);
   string str_r1 = (string)("pitch: ")+ to_string(-pitch_angle*180/3.14);
   string str_r2 = (string)("yaw: ")+ to_string(-yaw_angle*180/3.14);
   putText(_img, str_d, Point(150, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
   putText(_img, str_r1, Point(10, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
   putText(_img, str_r2, Point(310, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));

   return tVec;
}

Mat Armordetection::get_num_img(int img_size, bool show_num_img)
{
   Light light1 = _target_two_light.at(0);
   Light light2 = _target_two_light.at(1);

   // 保证light1是左灯条
   if(light1.get_center().x > light2.get_center().x)
   {
       light1 = Light(_target_two_light.at(1));
       light2 = Light(_target_two_light.at(0));
   }

   Mat Matrix, num_img;
   Point2f sourcePoints[4], objectPoints[4];

   double k1 = light1.line_k();
   int light1_height = abs(light1.Top_point().y - light1.Bottom_point().y);
   Point  p1 = light1.Right_point() + _now_roi.tl();
   int y11 = light1.Top_point().y - 0.6 * light1_height + _now_roi.tl().y;
   int y12 = light1.Bottom_point().y + 0.6 * light1_height + _now_roi.tl().y;
   int x11 = (y11 - p1.y)/k1 + p1.x;
   int x12 = (y12 - p1.y)/k1 + p1.x;

   double k2 = light2.line_k();
   int light2_height = abs(light2.Top_point().y - light2.Bottom_point().y);
   Point  p2 = light2.Left_point()+ _now_roi.tl();
   int y21 = light2.Top_point().y - 0.6 * light2_height + _now_roi.tl().y;
   int y22 = light2.Bottom_point().y + 0.6 * light2_height + _now_roi.tl().y;
   int x21 = (y21 - p2.y)/k1 + p2.x;
   int x22 = (y22 - p2.y)/k1 + p2.x;

   // 保证以上四个点的坐标不会超出图片的范围
   x11 = max(1, x11);
   x11 = min(_img.cols, x11);
   y11 = max(1, y11);
   y11 = min(_img.rows, y11);

   x21 = max(1, x21);
   x21 = min(_img.cols, x21);
   y21 = max(1, y21);
   y21 = min(_img.rows, y21);

   x12 = max(1, x12);
   x12 = min(_img.cols, x12);
   y12 = max(1, y12);
   y12 = min(_img.rows, y12);

   x22 = max(1, x22);
   x22 = min(_img.cols, x22);
   y22 = max(1, y22);
   y22 = min(_img.rows, y22);

   sourcePoints[0].x = x11; sourcePoints[0].y = y11; //left_top
   sourcePoints[1].x = x21; sourcePoints[1].y = y21;  //right_top
   sourcePoints[2].x = x12; sourcePoints[2].y = y12;   //left_bottom
   sourcePoints[3].x = x22; sourcePoints[3].y = y22;  //right_bottom

   objectPoints[0].x = 0; objectPoints[0].y = 0;
   objectPoints[1].x = img_size; objectPoints[1].y = 0;
   objectPoints[2].x = 0; objectPoints[2].y = img_size;
   objectPoints[3].x = img_size; objectPoints[3].y = img_size;

   Matrix = getPerspectiveTransform(sourcePoints, objectPoints);
   warpPerspective(_frame, num_img, Matrix, Size(img_size, img_size), INTER_LINEAR);

  if(show_num_img){
       namedWindow("num_img", WINDOW_NORMAL);
       imshow("num_img", num_img);
       circle(_img, sourcePoints[0], 4, Scalar(0,0,255), -1);
       circle(_img, sourcePoints[1], 4, Scalar(0,0,255), -1);
       circle(_img, sourcePoints[2], 4, Scalar(0,0,255), -1);
       circle(_img, sourcePoints[3], 4, Scalar(0,0,255), -1);
   }

   return num_img;
}

void Armordetection::begin_to_detect(Mat c, Mat d, double roi_scale, double armorH, double armorW,
                     bool use_roi, int color_detection_mode, bool show_mask,
                     bool show_light_filter_result, bool show_match_result,
                     bool get_num_image, int num_img_size, bool show_num_img)
{
    double t = (double)getTickCount();
    vector<vector<Point>> light_filter, match;
    vector<Point> target;

    this->set_img(use_roi);
    this->color_detection(color_detection_mode, show_mask);

    light_filter = this->light_filter(show_light_filter_result);
    match = this->light_match(light_filter, show_match_result);

    // 如果有识别到目标
    if(match.size()>=2){
     _flag = 1;     // 目前先用1来表示检测到装甲板，等数字识别搞定了，再细分，用1，2区分大小装甲板
    target = this->choose_target(match);  // 打击目标的四个顶点
    _rotaterect = minAreaRect(target);
    _next_roi = this->get_roi(target, roi_scale);
    rectangle(_img, _now_roi, Scalar(0, 255, 0), 2, LINE_AA, 0);
    _tVec = this->get_tVec(target, c, d, armorH, armorW);
    if(get_num_image){
       _num_img = this->get_num_img(num_img_size, show_num_img);
     }
    }
    else
    {
        rectangle(_img, _now_roi, Scalar(0, 255, 0), 2, LINE_AA, 0);
        if (_lossFrameK < 10) {

            if (_lossFrameK> 0)
            {
                roi_scale = 0.5*roi_scale + 0.05 * (double)_lossFrameK;
            }

             int xmin = _now_roi.tl().x;
             int ymin = _now_roi.tl().y;
             int xmax = _now_roi.br().x;
             int ymax = _now_roi.br().y;

              xmin = max(1, (int)(xmin - 2*(xmax-xmin)*roi_scale));
              ymin = max(1, (int)(ymin - 2*(ymax-ymin)*roi_scale));

              int rw = min((int)(xmax - xmin +(xmax-xmin)*roi_scale), _img.cols-xmin);
              rw = max(1, rw);
              int rh = min((int)(ymax - ymin +(ymax-ymin)*roi_scale), _img.rows-ymin);
              rh = max(1, rh);

               _next_roi = Rect(xmin, ymin, rw, rh);
          }

        else
         {
              _next_roi = Rect(0, 0, _img.cols, _img.rows);
         }
    }

//    // 计算帧率
//    _dt = ((double)getTickCount() - t) / (double)getTickFrequency();
//    _fps = 1/_dt;
//    string fpsString = (string)("fps: ")+ to_string(int(_fps + 0.5));
//    putText(_img, fpsString, Point(20, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
}

