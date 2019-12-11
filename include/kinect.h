/**
 * kinect.h  V 1.0 
 * A driver for Kinect V2 running on ubuntu 16.04
 * If you want using kinect.h to into your project, please 
 * move kinect.h to "{workspaceFolder}/include" 
 * add set(INCLUDE "${CMAKE_SOURCE_DIR}/include") in CMakeLists.txt
 * add include_directories(${INCLUDE}) in CMakeLists.txt
 * 
 * @Author zyli
 * @Time   2019.12.11
 * */

#pragma once

// C++ headers
#include <iostream>
#include <string>
#include <signal.h> 
#include <cstdlib>
#include <chrono> 
// opencv headers 
#include <opencv2/opencv.hpp>
// kinect driver libfreenect2 headers 
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/logger.h>
// PCL headers
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>


// what type of processor
enum Processor{
	CPU, OPENCL, OPENGL, CUDA
};
bool stop = false; //stop 标志位

void sigint_handler(int s){
	stop = true;
}

class Kinect {

public:
    //Create and initiate
    Kinect(Processor freenectprocessor = CPU) : listener_(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth), 
	                                       undistorted_(512, 424, 4), registered_(512, 424, 4), big_mat_(1920, 1082, 4), qnan_(std::numeric_limits<float>::quiet_NaN()){
        signal(SIGINT,sigint_handler);

        if(freenect2_.enumerateDevices() == 0)
        {
            std::cout << "no kinect2 connected!" << std::endl;
            exit(-1);
        }

        switch (freenectprocessor)
        {
            case CPU:
                std::cout << "creating Cpu processor" << std::endl;
                dev_ = freenect2_.openDefaultDevice (new libfreenect2::CpuPacketPipeline ());
                std::cout << "created" << std::endl;
                break;
            case OPENGL:
                std::cout << "creating OpenGL processor" << std::endl;
                dev_ = freenect2_.openDefaultDevice (new libfreenect2::OpenGLPacketPipeline ());
                break;
            default:
                std::cout << "creating Cpu processor" << std::endl;
                dev_ = freenect2_.openDefaultDevice (new libfreenect2::CpuPacketPipeline ());
                break;
        }

        dev_->setColorFrameListener(&listener_);
        dev_->setIrAndDepthFrameListener(&listener_);
        dev_->start();

        logger_ = libfreenect2::getGlobalLogger();
        registration_ = new libfreenect2::Registration(dev_->getIrCameraParams(), dev_->getColorCameraParams());

         prepareMake3D(dev_->getIrCameraParams());
    }

    //Stop and close 
    void shutDown(){
		dev_->stop();
  		dev_->close();
	}
    
	// Only color
	void getColor(cv::Mat & color_mat){
		listener_.waitForNewFrame(frames_);
		libfreenect2::Frame * rgb = frames_[libfreenect2::Frame::Color];

		cv::Mat tmp_color(rgb->height, rgb->width, CV_8UC4, rgb->data);

		tmp_color.copyTo(color_mat);
		// cv::flip(tmp_color, color_mat, 1);

		listener_.release(frames_);
	}

    // Only depth
	void getDepth(cv::Mat depth_mat){
		listener_.waitForNewFrame(frames_);
		libfreenect2::Frame * depth = frames_[libfreenect2::Frame::Depth];
		
		cv::Mat tmp_depth(depth->height, depth->width, CV_32FC1, depth->data);
        
		tmp_depth.copyTo(depth_mat);
		// cv::flip(tmp_depth, depth_mat, 1);

		listener_.release(frames_);
	}

	// Depth and color and map
	void get(cv::Mat & color_mat, cv::Mat & depth_mat, int* color_depth_map=0){
		listener_.waitForNewFrame(frames_);
		libfreenect2::Frame * rgb = frames_[libfreenect2::Frame::Color];
		libfreenect2::Frame * depth = frames_[libfreenect2::Frame::Depth];

		registration_->apply(rgb, depth, &undistorted_, &registered_, false, &big_mat_, color_depth_map);

		cv::Mat tmp_depth(undistorted_.height, undistorted_.width, CV_32FC1, undistorted_.data);
		cv::Mat tmp_color(rgb->height, rgb->width, CV_8UC4, rgb->data);
		
		tmp_depth.copyTo(depth_mat);
		tmp_color.copyTo(color_mat);
		// cv::flip(tmp_depth, depth_mat, 1);
		// cv::flip(tmp_color, color_mat, 1);

		listener_.release(frames_);
	}

	// Depth and color are aligned and registered 
	void getImage(cv::Mat & color_mat, cv::Mat & depth_mat, cv::Mat & ir_mat, const bool full_hd = true, const bool remove_points = false){
		listener_.waitForNewFrame(frames_);
		libfreenect2::Frame * rgb = frames_[libfreenect2::Frame::Color];
		libfreenect2::Frame * depth = frames_[libfreenect2::Frame::Depth];
		libfreenect2::Frame * ir = frames_[libfreenect2::Frame::Ir];

		registration_->apply(rgb, depth, &undistorted_, &registered_, remove_points, &big_mat_, map_);

		cv::Mat tmp_depth(undistorted_.height, undistorted_.width, CV_32FC1, undistorted_.data);
		cv::Mat tmp_color;
		cv::Mat ir_tmp(ir->height, ir->width, CV_32FC1, ir->data);

		if(full_hd)
			tmp_color = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);
		else
			tmp_color = cv::Mat(registered_.height, registered_.width, CV_8UC4, registered_.data);

		color_mat = tmp_color.clone();
		depth_mat = tmp_depth.clone();
		ir_mat = ir_tmp.clone();

		listener_.release(frames_);
	}

    // Only point cloud
    void getCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud){
        listener_.waitForNewFrame(frames_);
		libfreenect2::Frame * rgb = frames_[libfreenect2::Frame::Color];
		libfreenect2::Frame * depth = frames_[libfreenect2::Frame::Depth];

		registration_->apply(rgb, depth, &undistorted_, &registered_, true, &big_mat_, map_);
		const std::size_t w = undistorted_.width;
		const std::size_t h = undistorted_.height;

        cv::Mat tmp_itD0(undistorted_.height, undistorted_.width, CV_8UC4, undistorted_.data);
        cv::Mat tmp_itRGB0(registered_.height, registered_.width, CV_8UC4, registered_.data);

        // cv::flip(tmp_itD0,tmp_itD0,-1);
        // cv::flip(tmp_itRGB0,tmp_itRGB0,-1);

        const float * itD0 = (float *) tmp_itD0.ptr();
        const char * itRGB0 = (char *) tmp_itRGB0.ptr();
        
		pcl::PointXYZRGBA * itP = &cloud->points[0];
        bool is_dense = true;
		
		for(std::size_t y = 0; y < h; ++y){

			const unsigned int offset = y * w;
			const float * itD = itD0 + offset;
			const char * itRGB = itRGB0 + offset * 4;
			const float dy = rowmap_(y);

			for(std::size_t x = 0; x < w; ++x, ++itP, ++itD, itRGB += 4 )
			{
				const float depth_value = *itD / 1000.0f;
				
				if(!std::isnan(depth_value) && !(std::abs(depth_value) < 0.0001)){
	
					const float rx = colmap_(x) * depth_value;
                	const float ry = dy * depth_value;               
					itP->z = depth_value;
					itP->x = rx;
					itP->y = ry;

					itP->b = itRGB[0];
					itP->g = itRGB[1];
					itP->r = itRGB[2];
					itP->a = itRGB[2];
				} else {
					itP->z = qnan_;
					itP->x = qnan_;
					itP->y = qnan_;

					itP->b = qnan_;
					itP->g = qnan_;
					itP->r = qnan_;
					itP->a = qnan_;
					is_dense = false;
 				}
			}
		}
		cloud->is_dense = is_dense;
		listener_.release(frames_);
    }

	// Depth and color and map and cloud
	void get(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, cv::Mat & color_mat, cv::Mat & depth_mat, int* color_depth_map=0){
		listener_.waitForNewFrame(frames_);
		libfreenect2::Frame * rgb = frames_[libfreenect2::Frame::Color];
		libfreenect2::Frame * depth = frames_[libfreenect2::Frame::Depth];

		registration_->apply(rgb, depth, &undistorted_, &registered_, true, &big_mat_, color_depth_map);

		cv::Mat tmp_depth(undistorted_.height, undistorted_.width, CV_32FC1, undistorted_.data);
		cv::Mat tmp_color(rgb->height, rgb->width, CV_8UC4, rgb->data);

		tmp_depth.copyTo(depth_mat);
		tmp_color.copyTo(color_mat);
		// cv::flip(tmp_depth, depth_mat, 1); // 反转会导致映射关系不对应
		// cv::flip(tmp_color, color_mat, 1);

		const std::size_t w = undistorted_.width;
		const std::size_t h = undistorted_.height;

        cv::Mat tmp_itD0(undistorted_.height, undistorted_.width, CV_8UC4, undistorted_.data);
        cv::Mat tmp_itRGB0(registered_.height, registered_.width, CV_8UC4, registered_.data);

        // cv::flip(tmp_itD0,tmp_itD0,1);
        // cv::flip(tmp_itRGB0,tmp_itRGB0,1);

        const float * itD0 = (float *) tmp_itD0.ptr();
        const char * itRGB0 = (char *) tmp_itRGB0.ptr();
        
		pcl::PointXYZRGBA * itP = &cloud->points[0];
        bool is_dense = true;
		
		for(std::size_t y = 0; y < h; ++y){

			const unsigned int offset = y * w;
			const float * itD = itD0 + offset;
			const char * itRGB = itRGB0 + offset * 4;
			const float dy = rowmap_(y);

			for(std::size_t x = 0; x < w; ++x, ++itP, ++itD, itRGB += 4 )
			{
				const float depth_value = *itD / 1000.0f;
				
				if(!std::isnan(depth_value) && !(std::abs(depth_value) < 0.0001)){
	
					const float rx = colmap_(x) * depth_value;
                	const float ry = dy * depth_value;               
					itP->z = depth_value;
					itP->x = rx;
					itP->y = ry;

					itP->b = itRGB[0];
					itP->g = itRGB[1];
					itP->r = itRGB[2];
					itP->a = itRGB[3];
				} else {
					itP->z = qnan_;
					itP->x = qnan_;
					itP->y = qnan_;

					itP->b = qnan_;
					itP->g = qnan_;
					itP->r = qnan_;
					itP->a = qnan_;
					is_dense = false;
 				}
			}
		}
		cloud->is_dense = is_dense;
		listener_.release(frames_);
	}

	// 将深度图转化一下，方便显示
	void convertDepth(cv::Mat & depth_mat, cv::Mat & depth_8bit)
	{
		//  convert from 16bit to 8bit
		for (int row = 0; row < depth_mat.rows; row++)
		{
			for (int col = 0; col < depth_mat.cols; col++)
			{
				depth_8bit.at<uchar>(row, col) = ((int)depth_mat.at<float>(row, col)) % 256;
			}
		}
	}

	// 点云到彩色图的坐标映射，输入点云的下标，输出对应彩色图的下标
	cv::Point cloudToColor(cv::Point src, int* map)
	{
		cv::Point dst;
		dst.x = map[src.y*512+src.x]%1920;
		dst.y = map[src.y*512+src.x]/1920;
		return dst;
	}

	// 彩色图到点云的坐标映射，输入彩色图的下标，输出对应的点云下标
	cv::Point colorToCloud(cv::Point src, int* map)
	{
		cv::Point dst;
		int num=0;
		int pos = src.y*1920+src.x;
		for(int i=0;i<512*424;i++){
			if(abs(map[i]-pos)<3){
				dst.x = i%512;
				dst.y = i/512;
				num++;
			}
		}
		std::cout<<"num:"<<num<<endl;

		return dst;
	}

private:

    void prepareMake3D(const libfreenect2::Freenect2Device::IrCameraParams & depth_p)
	{
		const int w = 512;
		const int h = 424;
	    float * pm1 = colmap_.data();
	    float * pm2 = rowmap_.data();
	    for(int i = 0; i < w; i++)
	    {
	        *pm1++ = (i-depth_p.cx + 0.5) / depth_p.fx;
	    }
	    for (int i = 0; i < h; i++)
	    {
	        *pm2++ = (i-depth_p.cy + 0.5) / depth_p.fy;
	    }
	}

    libfreenect2::Freenect2 freenect2_;
	libfreenect2::Freenect2Device * dev_ = 0;
	libfreenect2::PacketPipeline * pipeline_ = 0;
	libfreenect2::Registration * registration_ = 0;
	libfreenect2::SyncMultiFrameListener listener_;
	libfreenect2::Logger * logger_ = nullptr;
	libfreenect2::FrameMap frames_;
	libfreenect2::Frame undistorted_,registered_,big_mat_;
	Eigen::Matrix<float,512,1> colmap_;
	Eigen::Matrix<float,424,1> rowmap_;
	std::string serial_;
	int map_[512 * 424];
    float qnan_;

};
