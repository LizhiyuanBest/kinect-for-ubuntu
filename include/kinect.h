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
bool stop = false; //stop flag

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
    
	/** Get color image.
	 * @param[out] color_mat Color image (1920x1080 BGRX).
	 */ 
	void getColor(cv::Mat & color_mat){
		listener_.waitForNewFrame(frames_);
		libfreenect2::Frame * rgb = frames_[libfreenect2::Frame::Color];

		cv::Mat tmp_color(rgb->height, rgb->width, CV_8UC4, rgb->data);

		tmp_color.copyTo(color_mat);
		// cv::flip(tmp_color, color_mat, 1);

		listener_.release(frames_);
	}

	/** Get depth image.
	 * @param[out] depth_mat Depth image (512x424 float).
	 */ 
	void getDepth(cv::Mat depth_mat){
		listener_.waitForNewFrame(frames_);
		libfreenect2::Frame * depth = frames_[libfreenect2::Frame::Depth];
		
		cv::Mat tmp_depth(depth->height, depth->width, CV_32FC1, depth->data);
        
		tmp_depth.copyTo(depth_mat);
		// cv::flip(tmp_depth, depth_mat, 1);

		listener_.release(frames_);
	}

	/** Get depth image and color image and map for color pixel on depth image.
	 * @param[out] color_mat Color image (1920x1080 BGRX).
	 * @param[out] depth_mat Depth image (512x424 float).
	 * @param[out] color_depth_map Index of mapped color pixel for each depth pixel (512x424).
	 */ 
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

	/** Get depth image and color image and ir and map for color pixel on depth image.
	 * @param[out] color_mat Color image (1920x1080 BGRX).
	 * @param[out] depth_mat Depth image (512x424 float).
	 * @param[out] ir_mat Ir image (512x424 float).
	 * @param[out] color_depth_map Index of mapped color pixel for each depth pixel (512x424).
	 */ 
	void getImage(cv::Mat & color_mat, cv::Mat & depth_mat, cv::Mat & ir_mat,int *color_depth_map=0){
		listener_.waitForNewFrame(frames_);
		libfreenect2::Frame * rgb = frames_[libfreenect2::Frame::Color];
		libfreenect2::Frame * depth = frames_[libfreenect2::Frame::Depth];
		libfreenect2::Frame * ir = frames_[libfreenect2::Frame::Ir];

		registration_->apply(rgb, depth, &undistorted_, &registered_, false, &big_mat_, color_depth_map);

		cv::Mat tmp_depth(undistorted_.height, undistorted_.width, CV_32FC1, undistorted_.data);
		cv::Mat tmp_color(rgb->height, rgb->width, CV_8UC4, rgb->data);
		cv::Mat ir_tmp(ir->height, ir->width, CV_32FC1, ir->data);
		
		color_mat = tmp_color.clone();
		depth_mat = tmp_depth.clone();
		ir_mat = ir_tmp.clone();

		listener_.release(frames_);
	}

	/** Only get point cloud.
	 * @param[out] cloud Point cloud (512x424 XYZRGBA).
	 */ 
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

	/** Get depth image and color image and cloud and map for color pixel on depth image.
	 * @param[out] cloud Point cloud (512x424 XYZRGBA).
	 * @param[out] color_mat Color image (1920x1080 BGRX).
	 * @param[out] depth_mat Depth image (512x424 float).
	 * @param[out] color_depth_map Index of mapped color pixel for each depth pixel (512x424).
	 */ 
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

	/** Convert depth images from float to uchar, and intent to show clearly
	 * @param depth_mat Color image (512x424 float)
	 * @param[out] depth_8bit Depth image (512x424 uchar)
	 */	
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

	/** Map cloud onto color images 
	 * @param src Cloud's subscript
	 * @param map index of mapped color pixel for each depth pixel (512x424). It can be got by the function get(cloud, color, depth, map).
	 * [out] Point A point of mapped color's subscript.
	 */ 
	cv::Point cloudToColor(cv::Point src, int* map)
	{
		cv::Point dst;
		dst.x = map[src.y*512+src.x]%1920;
		dst.y = map[src.y*512+src.x]/1920;
		return dst;
	}

	/** Map color images onto point cloud 
	 * @param src Color's subscript
	 * @param map index of mapped color pixel for each depth pixel (512x424). It can be got by the function get(cloud, color, depth, map).
	 * [out] Point A point of mapped cloud's subscript.
	 */ 
	cv::Point colorToCloud(cv::Point src, int* map)
	{
		static cv::Point dst(0,0);
		int pos = src.y*1920+src.x;
		for(int i=0;i<512*424;i++){
			if((abs(map[i]-pos-1920)<2) || (abs(map[i]-pos)<2) || (abs(map[i]-pos+1920)<2)){
				dst.x = i%512;
				dst.y = i/512;
				return dst;
			}
		}
		
		return dst;
	}

private:

	/** Prepare make point cloud 
	 * @param depth_p IrCameraParams
	 */
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
