#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/core/utility.hpp>
#include<sstream>
#include<iostream>
#include<QDebug>
using std::vector;
using namespace cv;
using std::stringstream;
static Scalar randomColor(RNG& rng)
{
	int icolor = (unsigned)rng;
	return Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}
void FindContoursBasic(Mat img) {
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	// 参数：img输入图像 contours轮廓输出 Hierarchy保存轮廓层次结构（可选） Mode检索轮廓模式  Method检索轮廓方法
	findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);//RETR_EXTERNAL:仅检测外部轮廓 CHAIN_APPROX_SIMPLE 压缩所有水平、垂直、对角线、仅存储起终点
	Mat output = Mat::zeros(img.rows, img.cols, CV_8UC3);
	//检查轮廓数量 如检测对象数作用一致
	if (contours.size() == 0) {
        qDebug() << "没有对象被检测" ;
		return;
	}
	else {
        qDebug() << "被检测到的对象个数为：" << contours.size() ;
	}
	RNG rng(0xFFFFFFFF);
	for (auto i = 0; i < contours.size(); i++) {
        drawContours(output, contours, i, randomColor(rng));//绘制对象区域轮廓
		namedWindow("ContoursResult", WINDOW_KEEPRATIO);
		imshow("ContoursResult", output);
	}
}

void ConnectedComponentsStats(Mat img2,Mat img)
{
    Mat labels, stats, centroids;
    auto num_objects = connectedComponentsWithStats(img, labels, stats, centroids);
    if (num_objects < 2) {
        qDebug() << "没有对象被检测到" << endl;
        return;
    }
    else {
        qDebug() << "检测到的对象数为：" << num_objects - 1 << endl;//检测到的对象数需要减去背景
    }
    Mat output = Mat::zeros(img.rows, img.cols, CV_8UC3);
    //RNG rng(0xFFFFFFFF);
    //声明一个白色色值
    Scalar color = Scalar(255,255,255);
    for (int i = 1; i < num_objects; i++) {
        //qDebug()<< "Object" << i << "with pos:" << centroids.at<Point2d>(i) << "with area" << stats.at<int>(i, CC_STAT_AREA) ;//通过标签显示检测到的对象的相关信息
        //CC_STAT_AREA 连通域的面积 像素点数
        Mat mask = labels == i;
        output.setTo(color, mask);//每个连通区域随机用一种颜色表示

        //绘制信息展示的区域
        stringstream ss;
        ss << "area:" << stats.at<int>(i, CC_STAT_AREA);
        putText(output, ss.str(), centroids.at<Point2d>(i), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255));//putText()在图像中绘制文本信息  centroids为文本位置 FONT_HERSHEY_SIMPLEX字体
        if(stats.at<int>(i, CC_STAT_AREA)>10000){
            Mat img3;
            bitwise_and(output,img2,img3);//掩膜与原图进行与运算
            namedWindow("lastResult", WINDOW_KEEPRATIO);
            imshow("lastResult", img3);
        }
    }
    namedWindow("StatsResult", WINDOW_KEEPRATIO);
    imshow("StatsResult", output);

}
void ConnectedComponents(Mat img2,Mat img) {
    //调用connectedComponents函数 通过返回的检测到的对象数 判断检测情况 小于2即只检测到背景，不需要进行绘制
    Mat labels;//与输入图像大小相同的输出矩阵
    auto num_objects = connectedComponents(img, labels);
    if (num_objects < 2) {
        qDebug() << "没有对象被检测到" ;
        return;
    }
    else {
        qDebug() << "检测到的对象数为：" << num_objects - 1 ;//检测到的对象数需要减去背景
    }
    //在新图像中用不同颜色绘制检测到的对象
    //新建一个输入大小，三通道黑色图像
    Mat output = Mat::zeros(img.rows, img.cols, CV_8UC3);
    RNG rng(0xFFFFFFFF);
    //声明一个白色色值
    //Scalar color = Scalar(255,255,255);
    //遍历每一个非0标签
    for (int i = 1; i < num_objects; i++) {
        Mat mask = labels == i;
        output.setTo(randomColor(rng), mask);//将对象轮廓内的色素值置为白色
    }
    namedWindow("Result", WINDOW_KEEPRATIO);
    imshow("Result", output);



}
int main() {
	Mat img = imread("D:/3D/depthImage.png",IMREAD_GRAYSCALE);
	namedWindow("colorImage", WINDOW_KEEPRATIO);
	imshow("colorImage", img);

	Mat img_noise;
	medianBlur(img,img_noise,3);
	namedWindow("噪声消除", WINDOW_KEEPRATIO);
	imshow("噪声消除", img_noise); //灰度图片中观察像素值大概范围 决定阈值处理方式
	Mat img_thr, binary_img;
	threshold(img_noise, img_thr, 30, 255, THRESH_TOZERO_INV);// THRESH_TOZERO_INV  当前点值大于阈值时，设置为0，否则不改变
	//截断阈值化：设定某个值为阈值，比阈值大的值均设置为阈值大小，比阈值小的值保持不变
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5,5));
	erode(img_thr, binary_img, kernel);//通过腐蚀尽量让图像区域分开点
	//dilate(img_thr, img_thr2, kernel);
	//结果展示
	namedWindow("二值化", WINDOW_KEEPRATIO);
	imshow("二值化", img_thr);
	namedWindow("腐蚀后结果", WINDOW_KEEPRATIO);
	imshow("腐蚀后结果", binary_img);
	//轮廓绘制
	FindContoursBasic(binary_img);

    //彩色原图
    Mat color_img=imread("D:/3D/colorImage.png");
    //抠图
    ConnectedComponentsStats(color_img,binary_img);
    ConnectedComponents(color_img,binary_img);
	waitKey(0);
	return 0;
}

