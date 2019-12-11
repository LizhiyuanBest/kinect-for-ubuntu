
#include "kinect.h"

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>(512, 424));
cv::Mat color(1080, 1920, CV_8UC4);
cv::Mat depth(424, 512, CV_32FC1);
cv::Mat depth_8bit(424, 512, CV_8UC1);
int map[512 * 424]; // Index of mapped color pixel for each depth pixel (512x424).


int col = 200; //列
int row = 300; //行
cv::Point onePoint(col, row);

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

void viewerPsycho (pcl::visualization::PCLVisualizer& viewer)
{
    static int time=0;
    viewer.setBackgroundColor (0.1f, 0.1f, 0.1f);
    pcl::PointXYZ o;
    o.x = cloud->at(onePoint.x, onePoint.y).x;
    o.y = cloud->at(onePoint.x, onePoint.y).y;
    o.z = cloud->at(onePoint.x, onePoint.y).z;
    viewer.addSphere (o, 0.01, "sphere"+ std::to_string(time), 0);
    time++;
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
        //验证对应关系 depth -> color -> cloud
        // cloud Point(200,300) -> color Point(map[300*512+200]%1920,map[300*512+200]/1920) -> depth Point(200,300)
        cv::circle(depth_8bit, onePoint, 10, cv::Scalar(0, 0, 255));
        cv::Point mapPoint = kinect.cloudToColor(onePoint,map);
        cv::circle(color, mapPoint, 20, cv::Scalar(0, 0, 255));
        std::cout<<kinect.colorToCloud(mapPoint,map)<<endl;
        cv::circle(depth_8bit, kinect.colorToCloud(mapPoint,map), 20, cv::Scalar(0, 0, 255));
        // static int show = 0;
        // if(show == 0){
        //     show ++;
        //     viewer->runOnVisualizationThreadOnce (viewerPsycho);
        // }
        viewer->runOnVisualizationThreadOnce (viewerPsycho);

        // std::cout<<"map: "<<map[300*512+200]<<" color: "<<map[300*512+200]%1920<<" "<<map[300*512+200]/1920<<"   ";
        std::cout<<"cloud: "<<cloud->at(onePoint.x, onePoint.y)<<"  depth: "<<depth.at<float>(onePoint.y,onePoint.x)<<"   ";
        std::cout<<"color: "<<color.at<cv::Vec4b>(mapPoint.y,mapPoint.x)<<endl;

        cv::imshow("color", color);
        cv::imshow("depth", depth_8bit);
        viewer->showCloud(cloud);//显示点云
        
        onePoint.x++;
        onePoint.y++;
        if(onePoint.x>=490 || onePoint.y>=400){
            onePoint.x = 30;
            onePoint.y =50;
        }

        int key = cv::waitKey(1);
        stop = stop || (key > 0 && ((key & 0xFF) == 27));
      
    }

    kinect.shutDown();
    
    return 0;
}

