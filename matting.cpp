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
void Matting(Mat img1,Mat img2){
    //声明标签 统计 形心 降噪后图像 灰度图像 阈值化后图像 腐蚀后图像
    Mat labels,stats,centroids,img_noise,gray_img,img_thr,img_ero;
    //对深度图像进行降噪处理 并存储到img_noise中
    medianBlur(img1,img_noise,3);
    //将图像转为灰度图像
    cvtColor(img_noise, gray_img, COLOR_BGR2GRAY);
    //对处理后的图像进行阈值处理
    //大于阈值时设置为0，否则不变
    threshold(gray_img,img_thr,80,255,THRESH_TOZERO_INV);

    //对阈值化后的图像进行腐蚀，使不同对象尽可能分离
    //Mat kernel = getStructuringElement(MORPH_RECT, Size(5,5));
    //erode(img_thr, img_ero, kernel);//通过腐蚀尽量让图像区域分开点
    //对腐蚀后的图像进行对象检测和轮廓绘制

    //对象检测，返回检测到的对象数
    auto num_objects=connectedComponents(img_thr,labels);
    //在新图像中用不同颜色绘制检测到的对象
    //新建一个输入大小，三通道黑色图像
    Mat output = Mat::zeros(img_thr.rows, img_thr.cols, CV_8UC3);
    //声明一个白色色值
    Scalar color = Scalar(255,255,255);
    //遍历每一个非0标签
    for (int i = 1; i < num_objects; i++) {
        //每个对象对应一个标签
        Mat mask = labels == i;
        //将对象轮廓内的色素值置为白色
        output.setTo(color, mask);

    }//此时的output图像可以作为掩膜去与彩色原图进行与运算进行抠图
    //最终结果图的声明.
    Mat result_img;
    //掩膜与原图进行与运算
    bitwise_and(output,img2,result_img);
    //抠图结果展示
    namedWindow("result_img", WINDOW_KEEPRATIO);
    imshow("result_img", result_img);

    //轮廓绘制部分
    vector<vector<Point>> contours; //声明二维浮点型向量 记录轮廓数据
    vector<Vec4i> hierarchy; //存放四维 int 变量 描绘层次结构
    // 参数：img输入图像 contours轮廓输出 Hierarchy保存轮廓层次结构（可选） Mode检索轮廓模式  Method检索轮廓方法
    //RETR_EXTERNAL:仅检测外部轮廓 CHAIN_APPROX_SIMPLE 压缩所有水平、垂直、对角线、仅存储起终点
    findContours(img_thr, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //一个用于绘制轮廓图的 与原图像大小一致的黑的图像
    Mat output2 = Mat::zeros(img2.rows, img2.cols, CV_8UC3);
    for(auto i =0;i< contours.size();i++){
        //绘制轮廓
        drawContours(output2,contours,i,Scalar(255,0,0));
        //轮廓图显示
        namedWindow("ContoursResult", WINDOW_KEEPRATIO);
        imshow("ContoursResult", output2);

    }



}
int main(){
    //读取深度图片
    Mat depth_img=imread("D:/20200711142344829/depthImage.png");
    //图片展示
    namedWindow("深度图像", WINDOW_KEEPRATIO);
    imshow("深度图像", depth_img);
    //读取彩色原图
    Mat color_img=imread("D:/20200711142344829/colorImage.png");
    //图片展示
    namedWindow("彩色原图", WINDOW_KEEPRATIO);
    imshow("彩色原图", color_img);
    Matting(depth_img,color_img);
    waitKey(0);
    return 0;
}
