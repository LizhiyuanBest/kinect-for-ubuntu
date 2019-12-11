
#include "kinect.h"

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>(512, 424));
cv::Mat color(1080, 1920, CV_8UC4);
cv::Mat depth(424, 512, CV_32FC1);
cv::Mat depth_8bit(424, 512, CV_8UC1);
int map[512 * 424]; // Index of mapped color pixel for each depth pixel (512x424).


//回调函数，当键盘有输入时，被调用
void keyboardEventOccured(const pcl::visualization::KeyboardEvent &event, void *nothing){
    static int time=0;
    if (event.getKeySym() == "space" && event.keyDown()){//当按下空格键时
        cout << "Space is pressed => pointcloud saved as output.pcd" << endl;
        pcl::io::savePCDFile("cloud" + std::to_string(time) + ".pcd", *cloud);
        cv::imwrite("color" + std::to_string(time) + ".bmp", color);
        cv::imwrite("depth" + std::to_string(time) + ".bmp", depth);
        time++;
    }
}

int main(int argc, char * argv[]){

    //create and initiate kinect
    Kinect kinect(OPENGL);
    //创建一个显示点云的窗口    
    boost::shared_ptr<pcl::visualization::CloudViewer> viewer (new pcl::visualization::CloudViewer ("viewer"));
    //绑定可视化窗口和键盘事件的函数
    viewer->registerKeyboardCallback(keyboardEventOccured,(void*)NULL); 

    while(!stop){
        kinect.get(cloud, color, depth, map);
        kinect.convertDepth(depth, depth_8bit);

        cv::imshow("color", color);
        cv::imshow("depth", depth_8bit);
        viewer->showCloud(cloud);//显示点云

        int key = cv::waitKey(1);
        stop = stop || (key > 0 && ((key & 0xFF) == 27));
      
    }
    kinect.shutDown();
    return 0;
}

