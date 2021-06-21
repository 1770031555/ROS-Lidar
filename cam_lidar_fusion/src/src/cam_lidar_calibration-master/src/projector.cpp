// Press 'i' to take a sample, Press 'o' to start optimization, Press 'e' to terminate node

#include "projector.h"
#include <iostream>

using namespace std;

//const darknet_ros_msgs::BoundingBoxesConstPtr& bd_boxes
std::vector<darknet_ros_msgs::BoundingBoxes::Ptr> bd_boxes_vec;
//std::vector<string> bd_boxes_vec;



void projector::undistort_img(cv::Mat original_img, cv::Mat undistort_img) {
    remap(original_img, undistort_img, undistort_map1, undistort_map2, cv::INTER_LINEAR);
}

void projector::init_params() {
    std::string pkg_loc = ros::package::getPath("cam_lidar_calibration");
    std::ifstream infile(pkg_loc + "/cfg/initial_params.txt");
    int cb_l, cb_b, l, b, e_l, e_b, i_l, i_b;
    infile >> i_params.camera_topic;
    infile >> i_params.lidar_topic;
    infile >> i_params.fisheye_model;
    infile >> i_params.lidar_ring_count;
    infile >> cb_l;
    infile >> cb_b;
    i_params.grid_size = std::make_pair(cb_l, cb_b);
    infile >> i_params.square_length;
    infile >> l;
    infile >> b;
    i_params.board_dimension = std::make_pair(l, b);
    infile >> e_l;
    infile >> e_b;
    i_params.cb_translation_error = std::make_pair(e_l, e_b);
    double camera_mat[9];
    for (int i = 0; i < 9; i++) {
        infile >> camera_mat[i];
    }
    cv::Mat(3, 3, CV_64F, &camera_mat).copyTo(i_params.cameramat);

    infile >> i_params.distcoeff_num;
    double dist_coeff[i_params.distcoeff_num];
    for (int i = 0; i < i_params.distcoeff_num; i++) {
        infile >> dist_coeff[i];
    }
    cv::Mat(1, i_params.distcoeff_num, CV_64F, &dist_coeff).copyTo(i_params.distcoeff);
    infile >> i_l;
    infile >> i_b;
    i_params.image_size = std::make_pair(i_l, i_b);

    // load image undistort params and get the re-map param
    // 去畸变并保留最大图
    cv::Size img_size(i_params.image_size.first, i_params.image_size.second);
    cv::initUndistortRectifyMap(i_params.cameramat, i_params.distcoeff, cv::Mat(),
                                cv::getOptimalNewCameraMatrix(i_params.cameramat, i_params.distcoeff, img_size, 1,
                                                              img_size, 0),
                                img_size, CV_16SC2, undistort_map1, undistort_map2);
}

double *projector::converto_imgpts(double x, double y, double z) {
    double tmpxC = x / z;
    double tmpyC = y / z;
    cv::Point2d planepointsC;

    planepointsC.x = tmpxC;
    planepointsC.y = tmpyC;

    double r2 = tmpxC * tmpxC + tmpyC * tmpyC;

    if (i_params.fisheye_model) {
        double r1 = pow(r2, 0.5);
        double a0 = std::atan(r1);
        double a1 = a0 * (1 + i_params.distcoeff.at<double>(0) * pow(a0, 2) +
                          i_params.distcoeff.at<double>(1) * pow(a0, 4)
                          + i_params.distcoeff.at<double>(2) * pow(a0, 6) +
                          i_params.distcoeff.at<double>(3) * pow(a0, 8));
        planepointsC.x = (a1 / r1) * tmpxC;
        planepointsC.y = (a1 / r1) * tmpyC;
        planepointsC.x = i_params.cameramat.at<double>(0, 0) * planepointsC.x + i_params.cameramat.at<double>(0, 2);
        planepointsC.y = i_params.cameramat.at<double>(1, 1) * planepointsC.y + i_params.cameramat.at<double>(1, 2);
    } else // For pinhole camera model
    {
        double tmpdist = 1 + i_params.distcoeff.at<double>(0) * r2 + i_params.distcoeff.at<double>(1) * r2 * r2 +
                         i_params.distcoeff.at<double>(4) * r2 * r2 * r2;
        planepointsC.x = tmpxC * tmpdist + 2 * i_params.distcoeff.at<double>(2) * tmpxC * tmpyC +
                         i_params.distcoeff.at<double>(3) * (r2 + 2 * tmpxC * tmpxC);
        planepointsC.y = tmpyC * tmpdist + i_params.distcoeff.at<double>(2) * (r2 + 2 * tmpyC * tmpyC) +
                         2 * i_params.distcoeff.at<double>(3) * tmpxC * tmpyC;
        planepointsC.x = i_params.cameramat.at<double>(0, 0) * planepointsC.x + i_params.cameramat.at<double>(0, 2);
        planepointsC.y = i_params.cameramat.at<double>(1, 1) * planepointsC.y + i_params.cameramat.at<double>(1, 2);
    }

    double *img_coord = new double[2];
    *(img_coord) = planepointsC.x;
    *(img_coord + 1) = planepointsC.y;

    return img_coord;
}

void projector::matrix_to_transfrom(Eigen::MatrixXf &matrix, tf::Transform &trans) {
    tf::Vector3 origin;
    origin.setValue(static_cast<double>(matrix(0, 3)), static_cast<double>(matrix(1, 3)),
                    static_cast<double>(matrix(2, 3)));

    tf::Matrix3x3 tf3d;
    tf3d.setValue(static_cast<double>(matrix(0, 0)), static_cast<double>(matrix(0, 1)),
                  static_cast<double>(matrix(0, 2)),
                  static_cast<double>(matrix(1, 0)), static_cast<double>(matrix(1, 1)),
                  static_cast<double>(matrix(1, 2)),
                  static_cast<double>(matrix(2, 0)), static_cast<double>(matrix(2, 1)),
                  static_cast<double>(matrix(2, 2)));

    tf::Quaternion tfqt;
    tf3d.getRotation(tfqt);

    trans.setOrigin(origin);
    trans.setRotation(tfqt);
}

void projector::sensor_info_callback_1(const sensor_msgs::Image::ConstPtr &img,
                                       const sensor_msgs::PointCloud2::ConstPtr &pc) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(img, "bgr8");
    }
    catch (cv_bridge::Exception &e) {
        return;
    }
//        cv::Mat ori_img = cv_ptr->image.clone();
//        undistort_img(ori_img, cv_ptr->image);

    cv::Mat raw_image = cv_ptr->image;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::fromROSMsg(*pc, *cloud);

    std::cout << "get pc and image data" << std::endl;

    cv::Mat new_image_raw;
    new_image_raw = raw_image.clone();

    //Extrinsic parameter: Transform Velodyne -> cameras
    tf::Matrix3x3 rot;
    rot.setRPY(rot_trans.roll, rot_trans.pitch, rot_trans.yaw);

    Eigen::MatrixXf t1(4, 4), t2(4, 4);
    t1 << rot.getRow(0)[0], rot.getRow(0)[1], rot.getRow(0)[2], rot_trans.x,
            rot.getRow(1)[0], rot.getRow(1)[1], rot.getRow(1)[2], rot_trans.y,
            rot.getRow(2)[0], rot.getRow(2)[1], rot.getRow(2)[2], rot_trans.z,
            0, 0, 0, 1;
    t2 = t1.inverse();

    tf::Transform lidar_to_cam, cam_to_lidar;
    matrix_to_transfrom(t1, cam_to_lidar);
    tf_br.sendTransform(tf::StampedTransform(cam_to_lidar, ros::Time::now(), "left_front", "camera"));

    Eigen::Affine3f transform_A = Eigen::Affine3f::Identity();
    transform_A.matrix() << t2(0, 0), t2(0, 1), t2(0, 2), t2(0, 3),
            t2(1, 0), t2(1, 1), t2(1, 2), t2(1, 3),
            t2(2, 0), t2(2, 1), t2(2, 2), t2(2, 3),
            t2(3, 0), t2(3, 1), t2(3, 2), t2(3, 3);

    pcl::PointCloud<pcl::PointXYZ>::Ptr organized(new pcl::PointCloud<pcl::PointXYZ>);
    organized_pointcloud(cloud, organized);

    for (pcl::PointCloud<pcl::PointXYZ>::const_iterator it = organized->points.begin();
         it != organized->points.end(); it++) {
        pcl::PointXYZ itA;
        itA = pcl::transformPoint(*it, transform_A);
        if (itA.z < 0 or std::abs(itA.x / itA.z) > 1.2)
            continue;

        double *img_pts = converto_imgpts(itA.x, itA.y, itA.z);
        double length = sqrt(pow(itA.x, 2) + pow(itA.y, 2) + pow(itA.z, 2)); //range of every point
        int color = std::min(round((length / 30) * 49), 49.0);

        if (img_pts[1] >= 0 and img_pts[1] < i_params.image_size.second
            and img_pts[0] >= 0 and img_pts[0] < i_params.image_size.first) {
            cv::circle(new_image_raw, cv::Point(img_pts[0], img_pts[1]), 3,
                       CV_RGB(255 * colmap[color][0], 255 * colmap[color][1], 255 * colmap[color][2]), -1);
        }
    }

    std::cout << "final images size: " << new_image_raw.size();

    // Publish the image projection
    ros::Time time = ros::Time::now();
    cv_ptr->encoding = "bgr8";
    cv_ptr->header.stamp = time;
    cv_ptr->header.frame_id = "/left_front";
    cv_ptr->image = new_image_raw;
    image_publisher.publish(cv_ptr->toImageMsg());

}


void projector::sensor_info_callback_2(const sensor_msgs::Image::ConstPtr &img,
                                       const sensor_msgs::PointCloud2::ConstPtr &pc) {
//                                       const darknet_ros_msgs::BoundingBoxesConstPtr &bd_boxes) {
//    ROS_INFO_STREAM("Hello This is pupil_darknet synCB...");
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(img, "bgr8");
    }
    catch (cv_bridge::Exception &e) {
        return;
    }
//        cv::Mat ori_img = cv_ptr->image.clone();
//        undistort_img(ori_img, cv_ptr->image);

    cv::Mat raw_image = cv_ptr->image;


    pcl::PointCloud<pcl::PointXYZ>::Ptr current_pc_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*pc, *current_pc_ptr);

//    std::cout << "get pc and image data" << std::endl;

    cv::Mat new_image_raw;
    new_image_raw = raw_image.clone();

    //Extrinsic parameter: Transform Velodyne -> cameras
    tf::Matrix3x3 rot;
    rot.setRPY(rot_trans.roll, rot_trans.pitch, rot_trans.yaw);

    Eigen::MatrixXf t1(4, 4), t2(4, 4);
    t1 << rot.getRow(0)[0], rot.getRow(0)[1], rot.getRow(0)[2], rot_trans.x,
            rot.getRow(1)[0], rot.getRow(1)[1], rot.getRow(1)[2], rot_trans.y,
            rot.getRow(2)[0], rot.getRow(2)[1], rot.getRow(2)[2], rot_trans.z,
            0, 0, 0, 1;
    t2 = t1.inverse();

    tf::Transform lidar_to_cam, cam_to_lidar;
    matrix_to_transfrom(t1, cam_to_lidar);
    tf_br.sendTransform(tf::StampedTransform(cam_to_lidar, ros::Time::now(), "left_front", "camera"));

    Eigen::Affine3f transform_A = Eigen::Affine3f::Identity();
    transform_A.matrix() << t2(0, 0), t2(0, 1), t2(0, 2), t2(0, 3),
            t2(1, 0), t2(1, 1), t2(1, 2), t2(1, 3),
            t2(2, 0), t2(2, 1), t2(2, 2), t2(2, 3),
            t2(3, 0), t2(3, 1), t2(3, 2), t2(3, 3);

    pcl::PointCloud<pcl::PointXYZ>::Ptr organized(new pcl::PointCloud<pcl::PointXYZ>);
    organized_pointcloud(current_pc_ptr, organized);

//    for (pcl::PointCloud<pcl::PointXYZ>::const_iterator it = organized->points.begin(); it != organized->points.end(); it++) {
//        pcl::PointXYZ itA;
//        itA = pcl::transformPoint(*it, transform_A);
//        if (itA.z < 0 or std::abs(itA.x / itA.z) > 1.2)
//            continue;
//
//        double *img_pts = converto_imgpts(itA.x, itA.y, itA.z);
//        double length = sqrt(pow(itA.x, 2) + pow(itA.y, 2) + pow(itA.z, 2)); //range of every point
//        int color = std::min(round((length / 30) * 49), 49.0);
//
//        if (img_pts[1] >= 0 and img_pts[1] < i_params.image_size.second
//            and img_pts[0] >= 0 and img_pts[0] < i_params.image_size.first) {
//            cv::circle(new_image_raw, cv::Point(img_pts[0], img_pts[1]), 3,
//                       CV_RGB(255 * colmap[color][0], 255 * colmap[color][1], 255 * colmap[color][2]), -1);
//        }
//    }

    // TODO
    if (bd_boxes_vec.size() > 0) {
        darknet_ros_msgs::BoundingBoxes::Ptr &bd_boxes_ = bd_boxes_vec.back();
        bd_boxes_vec.pop_back();

        for (auto bd_box : bd_boxes_->bounding_boxes) {
//        string bdbox_xmax = to_string(bd_box.xmax);
//        string bdbox_ymax = to_string(bd_box.ymax);
            float x_max_mine = -10000;
            float x_min_mine = +10000;
            float y_max_mine = -10000;
            float y_min_mine = +10000;
            float z_max_mine = -10000;
            float z_min_mine = +10000;

            for (pcl::PointCloud<pcl::PointXYZ>::const_iterator it = organized->points.begin();
                 it != organized->points.end(); it++) {
                pcl::PointXYZ itA;
                itA = pcl::transformPoint(*it, transform_A);
                if (itA.z < 0 or std::abs(itA.x / itA.z) > 1.2) {
                    continue;
                }
                double *img_pts = converto_imgpts(itA.x, itA.y, itA.z);
                double length = sqrt(pow(itA.x, 2) + pow(itA.y, 2) + pow(itA.z, 2)); //range of every point
                int color = std::min(round((length / 30) * 49), 49.0);


                if (img_pts[0] < bd_box.xmax && img_pts[0] > bd_box.xmin && img_pts[1] < bd_box.ymax && img_pts[1] > bd_box.ymin){

                    cv::circle(new_image_raw, cv::Point(img_pts[0], img_pts[1]), 3,
                               CV_RGB(255 * colmap[color][0], 255 * colmap[color][1], 255 * colmap[color][2]), -1);

                    if (itA.x>x_max_mine) x_max_mine=itA.x;
                    if (itA.x<x_min_mine) x_min_mine=itA.x;
                    if (itA.y>y_max_mine) y_max_mine=itA.y;
                    if (itA.y<y_min_mine) y_min_mine=itA.y;
                    if (itA.z>z_max_mine) z_max_mine=itA.z;
                    if (itA.z<z_min_mine) z_min_mine=itA.z;
                }
            }
//            char loca_str[30];
//            sprintf(loca_str, "%.2f, %.2f", bd_box.xmax, bd_box.ymax);
            cv::rectangle(new_image_raw, cv::Point(bd_box.xmax, bd_box.ymax), cv::Point(bd_box.xmax, bd_box.ymax),
                          cv::Scalar(00, 00, 255), 5);
            cv::putText(new_image_raw, bd_box.Class + "x:" + to_string(x_max_mine) + "y:" + to_string(y_max_mine) + "z:" + to_string(z_max_mine),
                        cv::Point(bd_box.xmin + 5, bd_box.ymin - 60), cv::FONT_HERSHEY_TRIPLEX, 1.3, cv::Scalar(0, 0, 255));
        }
    }


    std::cout << "final images size: " << new_image_raw.size();

    // Publish the image projection
    ros::Time time = ros::Time::now();
    cv_ptr->encoding = "bgr8";
    cv_ptr->header.stamp = time;
    cv_ptr->header.frame_id = "/left_front";
    cv_ptr->image = new_image_raw;
    image_publisher.publish(cv_ptr->toImageMsg());

}

void projector::sensor_info_callback_3(const sensor_msgs::Image::ConstPtr &img,
                                       const darknet_ros_msgs::BoundingBoxes::ConstPtr &msg) {
    ROS_INFO_STREAM("start : >>>>>");
    cout << "Bouding Boxes (header):" << msg->header << endl;
    cout << "Bouding Boxes (image_header):" << msg->image_header << endl;
    cout << "Bouding Boxes (Class):" << msg->bounding_boxes[0].Class << endl;
    cout << "Bouding Boxes (xmin):" << msg->bounding_boxes[0].xmin << endl;
    cout << "Bouding Boxes (xmax):" << msg->bounding_boxes[0].xmax << endl;
    cout << "Bouding Boxes (ymin):" << msg->bounding_boxes[0].ymin << endl;
    cout << "Bouding Boxes (ymax):" << msg->bounding_boxes[0].ymax << endl;
    cout << "\033[2J\033[1;1H";     // clear terminal
}


void projector::organized_pointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input_pointcloud,
                                     pcl::PointCloud<pcl::PointXYZ>::Ptr organized_pc) {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

    // Kdtree to sort the point cloud
    kdtree.setInputCloud(input_pointcloud);

    pcl::PointXYZ searchPoint;// camera position as target
    searchPoint.x = 0.0f;
    searchPoint.y = 0.0f;
    searchPoint.z = 0.0f;

    int K = input_pointcloud->points.size();
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);

    // Sort the point cloud based on distance to the camera
    if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
        for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i) {
            pcl::PointXYZ point;
            point.x = input_pointcloud->points[pointIdxNKNSearch[i]].x;
            point.y = input_pointcloud->points[pointIdxNKNSearch[i]].y;
            point.z = input_pointcloud->points[pointIdxNKNSearch[i]].z;
//            point.intensity = input_pointcloud->points[pointIdxNKNSearch[i]].intensity;
//            point.ring = input_pointcloud->points[pointIdxNKNSearch[i]].ring;
            organized_pc->points.push_back(point);
        }
    }
}

void msgCallback(const darknet_ros_msgs::BoundingBoxes::ConstPtr &msg) {
    ROS_INFO_STREAM("start collect msg");
    cout << "Bouding Boxes (header):" << msg->header << endl;
    cout << "Bouding Boxes (image_header):" << msg->image_header << endl;
    cout << "Bouding Boxes (Class):" << msg->bounding_boxes[0].Class << endl;
    cout << "Bouding Boxes (xmin):" << msg->bounding_boxes[0].xmin << endl;
    cout << "Bouding Boxes (xmax):" << msg->bounding_boxes[0].xmax << endl;
    cout << "Bouding Boxes (ymin):" << msg->bounding_boxes[0].ymin << endl;
    cout << "Bouding Boxes (ymax):" << msg->bounding_boxes[0].ymax << endl;
    cout << "\033[2J\033[1;1H";     // clear terminal
}

void msgCallback_2(const sensor_msgs::Image::ConstPtr &img,
                   const darknet_ros_msgs::BoundingBoxes::ConstPtr &msg) {
    ROS_INFO_STREAM("start collect msg");
    cout << "Bouding Boxes (header):" << msg->header << endl;
    cout << "Bouding Boxes (image_header):" << msg->image_header << endl;
    cout << "Bouding Boxes (Class):" << msg->bounding_boxes[0].Class << endl;
    cout << "Bouding Boxes (xmin):" << msg->bounding_boxes[0].xmin << endl;
    cout << "Bouding Boxes (xmax):" << msg->bounding_boxes[0].xmax << endl;
    cout << "Bouding Boxes (ymin):" << msg->bounding_boxes[0].ymin << endl;
    cout << "Bouding Boxes (ymax):" << msg->bounding_boxes[0].ymax << endl;
    cout << "\033[2J\033[1;1H";     // clear terminal
}


void projector::sensor_info_callback_4(const sensor_msgs::Image::ConstPtr &img,
                                       const darknet_ros_msgs::BoundingBoxesConstPtr &msg
) {
    ROS_INFO_STREAM("start collect msg");
//    cout<<"Bouding Boxes (header):" << msg->header <<endl;
//    cout<<"Bouding Boxes (image_header):" << msg->image_header <<endl;
//    cout<<"Bouding Boxes (Class):" << msg->bounding_boxes[0].Class <<endl;
//    cout<<"Bouding Boxes (xmin):" << msg->bounding_boxes[0].xmin <<endl;
//    cout<<"Bouding Boxes (xmax):" << msg->bounding_boxes[0].xmax <<endl;
//    cout<<"Bouding Boxes (ymin):" << msg->bounding_boxes[0].ymin <<endl;
//    cout<<"Bouding Boxes (ymax):" << msg->bounding_boxes[0].ymax <<endl;
//    cout << "\033[2J\033[1;1H";     // clear terminal
//    projector::sensor_info_callback_2(*img, *pc, *msg);
}


void chatterCallback(const darknet_ros_msgs::BoundingBoxes::Ptr &bd_boxes) {
//    pcl::PointCloud<pcl::PointXYZ>::Ptr current_pc_ptr(new pcl::PointCloud <pcl::PointXYZ>);
//    pcl::fromROSMsg(*pc, *current_pc_ptr);
////    pcl::copyPointCloud(*current_pc_ptr, *pc_const);
//    pc_const.push_back(current_pc_ptr);
////    for(size_t i=0;i<current_pc_ptr->size();i++) {
//////        const_pc->points.push_back(current_pc_ptr->points[i]);
//////        (*const_pc).points.push_back(current_pc_ptr->points[i]);
////        pc_const->points.push_back(current_pc_ptr->points[i]);
////    }
//    std::vector<darknet_ros_msgs::BoundingBoxesConstPtr> bd_boxes_vec;

    bd_boxes_vec.push_back(bd_boxes);
}


projector::projector() {
    ros::NodeHandle nh("~");
    init_params();

    nh.param<double>("roll", rot_trans.roll, 0.0);
    nh.param<double>("pitch", rot_trans.pitch, 0.0);
    nh.param<double>("yaw", rot_trans.yaw, 0.0);
    nh.param<double>("x", rot_trans.x, 0.0);
    nh.param<double>("y", rot_trans.y, 0.0);
    nh.param<double>("z", rot_trans.z, 0.0);


    ros::Publisher project_img_pub;

    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, i_params.camera_topic, 10);
//    message_filters::Subscriber <sensor_msgs::Image> image_sub(nh, "/darknet_ros/detection_image", 20);
    message_filters::Subscriber<sensor_msgs::PointCloud2> pcl_sub(nh, i_params.lidar_topic, 10);
//    ros::Subscriber sub = nh.subscribe(i_params.lidar_topic, 20, chatterCallback);
//    message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> bd_box_sub_(nh, "/darknet_ros/bounding_boxes", 20);

    ros::Subscriber sub_bdbox = nh.subscribe("/darknet_ros/bounding_boxes", 20, chatterCallback);


//        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
//        message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(5), image_sub, pcl_sub);
//        sync.registerCallback(boost::bind(&projector::sensor_info_callback_1, this, _1, _2));


//        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2, darknet_ros_msgs::BoundingBoxes> MySyncPolicy;
//        message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_sub, pcl_sub, bd_box_sub_);
//        sync->registerCallback(boost::bind(&projector::sensor_info_callback_2, this, _1, _2, _3));



    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(20), image_sub, pcl_sub);
    sync.registerCallback(boost::bind(&projector::sensor_info_callback_2, this, _1, _2));


//    message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> bd_box_sub__(nh, "/darknet_ros/bounding_boxes", 5);
//
//    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, darknet_ros_msgs::BoundingBoxes> MySyncPolicy;
//    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(5), image_sub, bd_box_sub__);
//    sync.registerCallback(boost::bind(&projector::sensor_info_callback_3, this, _1, _2));


//    ros::Subscriber cood_sub = nh.subscribe("/darknet_ros/bounding_boxes",100,msgCallback);

//        ros::Subscriber cood_sub = nh.subscribe("/darknet_ros/bounding_boxes",100, projector::sensor_info_callback_3);


    image_transport::ImageTransport imageTransport(nh);
    image_publisher = imageTransport.advertise("/project_pc_image", 1);

    ros::spin();
}


int main(int argc, char **argv) {
    ros::init(argc, argv, "project_pc_image");
    projector pj;
    return 0;
}

