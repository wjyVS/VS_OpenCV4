#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/core.hpp>
#include<QDebug>
using namespace cv;
using namespace std;
static Scalar randomColor(RNG& rng)
{
    int icolor = (unsigned)rng;
    return Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

void FindContoursBasic(Mat img2) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    // 参数：img输入图像 contours轮廓输出 Hierarchy保存轮廓层次结构（可选） Mode检索轮廓模式  Method检索轮廓方法
    findContours(img2, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);//RETR_EXTERNAL:仅检测外部轮廓 CHAIN_APPROX_SIMPLE 压缩所有水平、垂直、对角线、仅存储起终点
    Mat output = Mat::zeros(img2.rows, img2.cols, CV_8UC3);
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
    //RNG rng(0xFFFFFFFF);
    //声明一个白色色值
    Scalar color = Scalar(255,255,255);
    //遍历每一个非0标签
    for (int i = 1; i < num_objects; i++) {
        Mat mask = labels == i;
        output.setTo(color, mask);//将对象轮廓内的色素值置为白色
    }
    namedWindow("Result", WINDOW_KEEPRATIO);
    imshow("Result", output);//用于抠图的掩膜图像

    Mat gray_output,img3;

    bitwise_and(output,img2,img3);//掩膜与原图进行与运算
    namedWindow("lastResult", WINDOW_KEEPRATIO);
    imshow("lastResult", img3);

}
int main() {
    Mat img = imread("D:/20200711142344829/depthImage.png");
	namedWindow("深度图像", WINDOW_KEEPRATIO);
	imshow("深度图像", img);
    //降噪
    Mat img_noise;
    medianBlur(img,img_noise,5);
    //灰化
    Mat gray_img;
    cvtColor(img_noise, gray_img, COLOR_BGR2GRAY);
//    namedWindow("gray_img", WINDOW_KEEPRATIO);
//    imshow("gray_img", gray_img);
    //二值化处理
    Mat img_binary;
    threshold(gray_img,img_binary,80,255,THRESH_TOZERO_INV);
    namedWindow("img_binary", WINDOW_KEEPRATIO);
    imshow("img_binary", img_binary);
    //轮廓提取
    FindContoursBasic(img_binary);
    //彩色原图
    Mat color_img =imread("D:/20200711142344829/colorImage.png");
    namedWindow("color_img", WINDOW_KEEPRATIO);
    imshow("color_img", color_img);
    //检测对象 绘制轮廓 生成掩膜 利用掩膜与原图与运算实现抠图
    ConnectedComponents(color_img,img_binary);








	waitKey(0);
	destroyAllWindows();
	return 0;
}
