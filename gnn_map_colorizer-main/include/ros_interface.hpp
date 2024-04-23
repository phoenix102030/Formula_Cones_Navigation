#ifndef ROS_GNN_COLORIZER_HPP_
#define ROS_GNN_COLORIZER_HPP_

#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/Header.h>
#include <fs_msgs/Cones.h>
#include <fs_msgs/Cone.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_datatypes.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <vector>
#include <string>
#include <math.h>
#include <chrono>

#include "gnn_colorizer.hpp"

class ROS_GNNColorizer {

public:
    ROS_GNNColorizer() : nh_("~") {
        nh_.getParam("/gnn_map_colorizer/model_path", model_path_);
        nh_.getParam("/gnn_map_colorizer/propagation_distance", propagation_distance_);
        nh_.getParam("/gnn_map_colorizer/max_prop_angle", max_prop_angle_);
        nh_.getParam("/gnn_map_colorizer/max_input_range", max_input_range_);
        nh_.getParam("/gnn_map_colorizer/max_track_width", max_track_width_);
        nh_.getParam("/gnn_map_colorizer/association_threshold", association_threshold_);
        nh_.getParam("/gnn_map_colorizer/max_steps", max_steps_);

        sub_cones_ = nh_.subscribe("input_map", 1, &ROS_GNNColorizer::cones_callback, this);
        sub_odom_ = nh_.subscribe("input_odom", 1, &ROS_GNNColorizer::odom_callback, this);

        ROS_INFO("[GNN Map Colorizer] Subscribing to: %s", sub_cones_.getTopic().c_str());

        pub_cones_ = nh_.advertise<fs_msgs::Cones>("output_map", 1);
        pub_markers_ = nh_.advertise<visualization_msgs::MarkerArray>("output_map_markers", 1);
        pub_car_marker_ = nh_.advertise<visualization_msgs::Marker>("car_marker", 1);

        // find the full path to the ros package
        std::string path = ros::package::getPath("gnn_map_colorizer");
        
        model_path_ = path + "/" + model_path_;

        ROS_INFO("[GNN Map Colorizer] Model path: %s", model_path_.c_str());

        gnn_ = GNNColorizer(model_path_, 
                            propagation_distance_, 
                            max_prop_angle_, 
                            max_input_range_,
                            max_track_width_, 
                            association_threshold_);
    }

private:
    GNNColorizer gnn_;
    ros::NodeHandle nh_;
    ros::Subscriber sub_odom_;
    ros::Subscriber sub_cones_;
    ros::Publisher pub_cones_;
    ros::Publisher pub_markers_;
    ros::Publisher pub_car_marker_;

    std::string model_path_;
    float propagation_distance_;
    float max_prop_angle_;
    float max_input_range_;
    float max_track_width_;
    float association_threshold_;
    int max_steps_;

    float ego_x_ = 0;
    float ego_y_ = 0;
    float ego_yaw_ = 0;

    void odom_callback(const nav_msgs::Odometry::ConstPtr& msg) {
        ego_x_ = msg->pose.pose.position.x;
        ego_y_ = msg->pose.pose.position.y;
        ego_yaw_ = tf::getYaw(msg->pose.pose.orientation);
    }

    void cones_callback(const fs_msgs::Cones::ConstPtr& msg) {
        std::vector<Cone> slam_map;
        for (fs_msgs::Cone cone_msg : msg->cones) {
            Cone cone(cone_msg.x, cone_msg.y, 0, cone_msg.color);
            slam_map.push_back(cone);
        }

        gnn_.colorize(slam_map, ego_x_, ego_y_, ego_yaw_, max_steps_);
        publish_cones(slam_map, msg->header);
        publish_markers(slam_map, msg->header);
        publish_car_marker(ego_x_, ego_y_, ego_yaw_, msg->header);
    }

    uint8_t determineColor(const Cone& cone) {
        if (cone.color == 3 || cone.color == 4) {
            return static_cast<uint8_t>(cone.color); // small orange or big orange detected earlier
        }
        else if (cone.color_score > 0) {
            return static_cast<uint8_t>(1); // blue
        }
        else if (cone.color_score < 0) {
            return static_cast<uint8_t>(2); // yellow
        }
        return static_cast<uint8_t>(0); // unknown
    }

    void publish_cones(const std::vector<Cone>& slam_map, const std_msgs::Header header) {
        fs_msgs::Cones cones_msg;
        cones_msg.header = header;
        for (const Cone& cone : slam_map) {
            fs_msgs::Cone cone_msg;
            cone_msg.x = cone.x;
            cone_msg.y = cone.y;
            cone_msg.color = determineColor(cone);
            cones_msg.cones.push_back(cone_msg);
        }
        pub_cones_.publish(cones_msg);
    }

    void publish_markers(const std::vector<Cone>& slam_map, const std_msgs::Header header) {
        visualization_msgs::MarkerArray markers_msg;
        int id = 0;
        for (const Cone& cone : slam_map) {
            visualization_msgs::Marker marker_msg;
            marker_msg.header = header;
            marker_msg.lifetime = ros::Duration(0.1);
            marker_msg.id = id; id++;
            marker_msg.type = visualization_msgs::Marker::CYLINDER;
            marker_msg.action = visualization_msgs::Marker::ADD;
            marker_msg.pose.position.x = cone.x;
            marker_msg.pose.position.y = cone.y;
            marker_msg.pose.position.z = 0.0;
            marker_msg.pose.orientation.x = 0.0;
            marker_msg.pose.orientation.y = 0.0;
            marker_msg.pose.orientation.z = 1.0;
            marker_msg.pose.orientation.w = 1.0;
            marker_msg.scale.x = 0.5;
            marker_msg.scale.y = 0.5;
            marker_msg.scale.z = 1.0;
            marker_msg.color.a = 1.0;
            int color = determineColor(cone);
            if (color == 1) { // blue
                marker_msg.color.r = 0.0;
                marker_msg.color.g = 0.0;
                marker_msg.color.b = 1.0;
            }
            else if (color == 2) { // yellow
                marker_msg.color.r = 1.0;
                marker_msg.color.g = 1.0;
                marker_msg.color.b = 0.0;
            }
            else if (color == 3) { // small orange
                marker_msg.color.r = 1.0;
                marker_msg.color.g = 0.5;
                marker_msg.color.b = 0.0;
            }
            else if (color == 4) { // big orange (publish as red)
                marker_msg.color.r = 1.0;
                marker_msg.color.g = 0.0;
                marker_msg.color.b = 0.0;
            } 
            else { // unknown
                marker_msg.color.r = 0.5;
                marker_msg.color.g = 0.5;
                marker_msg.color.b = 0.5;
            }
            markers_msg.markers.push_back(marker_msg);
        }
        pub_markers_.publish(markers_msg);
    }

    void publish_car_marker(float x, float y, float yaw, const std_msgs::Header header) {
        visualization_msgs::Marker marker_msg;
        marker_msg.header = header;
        marker_msg.lifetime = ros::Duration(0.1);
        marker_msg.id = 0;
        marker_msg.type = visualization_msgs::Marker::ARROW;
        marker_msg.action = visualization_msgs::Marker::ADD;
        marker_msg.pose.position.x = x;
        marker_msg.pose.position.y = y;
        marker_msg.pose.position.z = 0.0;
        marker_msg.pose.orientation = tf::createQuaternionMsgFromYaw(yaw);
        marker_msg.scale.x = 1.5;
        marker_msg.scale.y = 1;
        marker_msg.scale.z = 1;
        marker_msg.color.a = 1.0;
        marker_msg.color.r = 1.0;
        marker_msg.color.g = 0.0;
        marker_msg.color.b = 0.0;
        pub_car_marker_.publish(marker_msg);
    }
};

#endif // ROS_GNN_COLORIZER_HPP_