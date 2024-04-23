
#ifndef TRACK_GENERATOR_HPP_
#define TRACK_GENERATOR_HPP_

#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <vector>

#include "common.hpp"

class Track {

public:
    Track() {};

    void getAllCones(std::vector<Point>& cones) {
        cones.clear();
        cones.insert(cones.end(), left_cones_.begin(), left_cones_.end());
        cones.insert(cones.end(), right_cones_.begin(), right_cones_.end());
        cones.insert(cones.end(), false_positives_.begin(), false_positives_.end());
    }

    void getLeftCones(std::vector<Point>& left_cones) {
        left_cones = left_cones_;
    }

    void getRightCones(std::vector<Point>& right_cones) {
        right_cones = right_cones_;
    }

    void getCenterPoints(std::vector<Point>& center_points) {
        center_points = center_points_;
    }

    void getFalsePositives(std::vector<Point>& false_positives) {
        false_positives = false_positives_;
    }

    int getOriginIndex() {
        return origin_index_;
    }

    int getUseEvery() {
        return use_every_;
    }

    void buildRealisticFSTrack(float detection_prob, int max_false_positives) {
        dummy_spacing_ = 0.5;
        float mean_track_width = 4.5;
        int num_radial_segments = static_cast<int>(random_uniform(2.1, 5.1));
        float cone_spacing = random_uniform(3.0, 6.0);

        buildPerfectTrack(num_radial_segments, dummy_spacing_, mean_track_width);

        float driven_distance = random_uniform(0.0, full_track_length_ / 1.5);
        driveForward(driven_distance);        
        addNoise();
        downsampleTrack(cone_spacing);
        addFalsePositives(max_false_positives);
        removeRandomCones(detection_prob);
    }


    void renormalizeTrack(const float propagation_dist, const float max_prop_angle) {
        float current_propagation_angle = getPropagationAngle(propagation_dist);
        rotatePoints(center_points_, -current_propagation_angle);
        rotatePoints(left_cones_, -current_propagation_angle);
        rotatePoints(right_cones_, -current_propagation_angle);

        const float new_propagation_angle = random_uniform(-max_prop_angle, max_prop_angle);

        rotatePoints(center_points_, new_propagation_angle);
        rotatePoints(left_cones_, new_propagation_angle);
        rotatePoints(right_cones_, new_propagation_angle);
    }

    void buildPerfectTrackAngles(const int num_radial_segments, const float dummy_spacing, const float mean_track_width) {   
        dummy_spacing_ = dummy_spacing;

        // turn is either 1 or -1, 1 means left turn, -1 means right turn
        int turn_direction = random_uniform(0.0, 1.0) < 0.5 ? 1 : -1;

        std::vector<float> angles;
        for (int i = 0; i < num_radial_segments; i++) {
            angles.push_back(random_uniform(-M_PI / 100.0, M_PI / 100.0));
        }

        float track_width = random_uniform(3.0, 3.5);

        for (float angle : angles) {
            float segment_length = random_uniform(5.0, 15.0);
            turn_direction *= 1; // switch direction

            addRadialSegmentAngle(angle, segment_length, turn_direction);
        }
        addLeftRightCones(track_width);
    }



    void buildPerfectTrack(const int num_radial_segments, const float dummy_spacing, const float mean_track_width) {   
        dummy_spacing_ = dummy_spacing;

        // turn is either 1 or -1, 1 means left turn, -1 means right turn
        int turn_direction = random_uniform(0.0, 1.0) < 0.5 ? 1 : -1;

        std::vector<float> radiuses = generateRadiuses(num_radial_segments);
        float track_width = generateTrackWidth(mean_track_width, radiuses);

        for (float radius : radiuses) {
            float segment_length = random_uniform(2 * radius * M_PI / 5.0, 2 * radius * M_PI / 2.0);
            
            if (segment_length > 15) { segment_length = 15; }

            turn_direction *= -1; // switch direction

            addRadialSegment(radius, segment_length, turn_direction);
        }
        addLeftRightCones(track_width);
    }

    void driveForward(const float distance) {
        int num_center_points = center_points_.size();

        origin_index_ = static_cast<int>(distance / dummy_spacing_);

        Point start_point = center_points_[origin_index_];
        Point next_point = center_points_[origin_index_ + 1];

        float new_start_angle = atan2(next_point.y - start_point.y, next_point.x - start_point.x);

        // move the track so that the start point is (0, 0)
        for (int i = 0; i < center_points_.size(); i++) {
            center_points_[i] -= start_point;
            left_cones_[i] -= start_point;
            right_cones_[i] -= start_point;
        }

        rotatePoints(center_points_, -new_start_angle);
        rotatePoints(left_cones_, -new_start_angle);
        rotatePoints(right_cones_, -new_start_angle);
    }


    void addNoise() {
        float rotation_noise = random_uniform(-M_PI / 7, M_PI / 7);
        rotatePoints(center_points_, rotation_noise);
        rotatePoints(left_cones_, rotation_noise);
        rotatePoints(right_cones_, rotation_noise);

        // add noise to the cones
        float x_noise = random_uniform(-2.0, 2.0);
        float y_noise = random_uniform(-2.0, 2.0);
        for (int i = 0; i < left_cones_.size(); i++) {
            left_cones_[i].x += x_noise;
            right_cones_[i].x += x_noise;
            center_points_[i].x += x_noise;

            left_cones_[i].y += y_noise;
            right_cones_[i].y += y_noise;
            center_points_[i].y += y_noise;
        }

        float ind_noise_level = 0.3;
        for (int i = 0; i < left_cones_.size(); i++) {
            float x_noise_left = random_uniform(-ind_noise_level, ind_noise_level);
            float y_noise_left = random_uniform(-ind_noise_level, ind_noise_level);
            float x_noise_right = random_uniform(-ind_noise_level, ind_noise_level);
            float y_noise_right = random_uniform(-ind_noise_level, ind_noise_level);
            left_cones_[i] += Point(x_noise_left, y_noise_left);
            right_cones_[i] += Point(x_noise_right, y_noise_right);
        }
    }


    void downsampleTrack(const float new_cone_spacing) {
        Point last_center_point = center_points_[0];

        use_every_ = static_cast<int>(new_cone_spacing / dummy_spacing_);

        std::vector<Point> downsampled_left_cones;
        std::vector<Point> downsampled_right_cones;

        for (int i = 0; i < left_cones_.size(); i+=use_every_) {
            downsampled_left_cones.push_back(left_cones_[i]);
            downsampled_right_cones.push_back(right_cones_[i]);
        }

        left_cones_ = downsampled_left_cones;
        right_cones_ = downsampled_right_cones;
    }


    void addFalsePositives(const int max_false_positives) {
        const int num_fps = static_cast<int>(random_uniform(0.0f, static_cast<float>(max_false_positives)));

        for (int i = 0; i < num_fps; i++) {
            float x = random_uniform(-50.0, 50.0);
            float y = random_uniform(-50.0, 50.0);
            false_positives_.push_back(Point(x, y));
        }
    }

    void removeRandomCones(const float detection_probability) {
        std::vector<Point> new_left_cones;
        std::vector<Point> new_right_cones;

        for (int i = 0; i < left_cones_.size(); i++) {
            float p = random_uniform(0.0, 1.0);
            if (p < detection_probability) {
                new_left_cones.push_back(left_cones_[i]);
            }
        }
        for (int i = 0; i < right_cones_.size(); i++) {
            float p = random_uniform(0.0, 1.0);
            if (p < detection_probability) {
                new_right_cones.push_back(right_cones_[i]);
            }
        }

        left_cones_ = new_left_cones;
        right_cones_ = new_right_cones;
    }

    float getPropagationAngle(const float propagation_dist) {
        for (int i = origin_index_; i < center_points_.size() - 1; i++) {
            Point center_point = center_points_[i];
            if (center_point.norm() > propagation_dist) {
                Point prev_center_point = center_points_[i - 1];
                return atan2(center_point.y, center_point.x);
            }
        }
        return 0;
    }

private:
    std::vector<Point> left_cones_;
    std::vector<Point> right_cones_;
    std::vector<Point> center_points_;
    std::vector<Point> false_positives_;

    int origin_index_ = 0;
    float dummy_spacing_;
    float full_track_length_ = 0;
    int use_every_ = 1; // use every nth cone when downsampling

    float generateTrackWidth(float mean_track_width, std::vector<float> radiuses) {
        float min_radius = 1000;

        for (float radius : radiuses) {
            if (radius < min_radius) {
                min_radius = radius;
            }
        }

        float track_width = mean_track_width + random_uniform(-1.7, 1.7);

        if (track_width > min_radius) {
            track_width = min_radius / random_uniform(1.8, 2.2);
        }

        if (track_width < 2.5) { track_width = 2.5; }
        if (track_width > 6.5) { track_width = 6.5; }

        return track_width;
    }


    std::vector<float> generateRadiuses(int radial_segments){
        std::vector<float> radiuses;
        for (int i = 0; i < radial_segments; i++) {
            
            float radius;
            float p = random_uniform(0.0, 1.0);
            if (p < 0.7) {
                radius = random_uniform(4.0, 10.0); 
            } 
            else {
                radius = random_uniform(10.0, 50.0);
            }
            radiuses.push_back(radius);
        }
        return radiuses;
    }


    void addRadialSegment(const float radius, const float segment_length, const int turn_direction) {
        float current_global_angle;
        Point last_point;

        if (center_points_.size() >= 2) {
            Point diff = center_points_[center_points_.size() - 1] - center_points_[center_points_.size() - 2];
            current_global_angle = atan2(diff.y, diff.x);
            last_point = center_points_[center_points_.size() - 1];
        }
        else {
            current_global_angle = 0;
            last_point = Point(0, 0);
        }

        const int segment_iterations = static_cast<int>(segment_length / dummy_spacing_);

        const float y = turn_direction * pow(dummy_spacing_, 2) / radius;
        const float x = sqrt(pow(dummy_spacing_, 2) - pow(y, 2));
        const float local_angle = atan2(y, x);

        const Point local_point(x, y);

        for (int i = 0; i < segment_iterations; i++) {
            current_global_angle += local_angle;
            Point local_point_rotated = local_point.rotated(current_global_angle);
            Point new_point = last_point + local_point_rotated;
            center_points_.push_back(new_point);
            last_point = new_point;
        }

        full_track_length_ += segment_length;
    }


    void addRadialSegmentAngle(const float angle, const float segment_length, const int turn_direction) {
        float current_global_angle;
        Point last_point;

        if (center_points_.size() >= 2) {
            Point diff = center_points_[center_points_.size() - 1] - center_points_[center_points_.size() - 2];
            current_global_angle = atan2(diff.y, diff.x);
            last_point = center_points_[center_points_.size() - 1];
        }
        else {
            current_global_angle = 0;
            last_point = Point(0, 0);
        }

        const int segment_iterations = static_cast<int>(segment_length / dummy_spacing_);

        const Point local_point(turn_direction * dummy_spacing_ * cos(angle), turn_direction * dummy_spacing_ * sin(angle));

        for (int i = 0; i < segment_iterations; i++) {
            current_global_angle += angle;
            Point local_point_rotated = local_point.rotated(current_global_angle);
            Point new_point = last_point + local_point_rotated;
            center_points_.push_back(new_point);
            last_point = new_point;
        }

        full_track_length_ += segment_length;
    }


    void addLeftRightCones(const float track_width) {
        left_cones_.resize(center_points_.size());
        right_cones_.resize(center_points_.size());

        const Point cone_to_rotate(track_width / 2.0, 0);

        Point first_left_cone;
        Point first_right_cone;

        const float first_angle = atan2(center_points_[1].y - center_points_[0].y, center_points_[1].x - center_points_[0].x);

        //left_cones_[0] = center_points_[0] + cone_to_rotate.rotated(first_angle + M_PI_2);
        //right_cones_[0] = center_points_[0] + cone_to_rotate.rotated(first_angle - M_PI_2);
        
        for (int i = 1; i < center_points_.size() - 1; i++) {
            Point p1 = center_points_[i - 1];
            Point p2 = center_points_[i + 1];
            float angle = atan2(p2.y - p1.y, p2.x - p1.x);

            left_cones_[i] = center_points_[i] + cone_to_rotate.rotated(angle + M_PI_2);
            right_cones_[i] = center_points_[i] + cone_to_rotate.rotated(angle - M_PI_2);
        }

        const Point last_cone = center_points_[center_points_.size() - 1];
        const Point second_last_cone = center_points_[center_points_.size() - 2];
        const float last_angle = atan2(last_cone.y - second_last_cone.y, last_cone.x - second_last_cone.x);
        //left_cones_[center_points_.size() - 1] = center_points_[0] + cone_to_rotate.rotated(last_angle + M_PI_2);
        //right_cones_[center_points_.size() - 1] = center_points_[0] + cone_to_rotate.rotated(last_angle - M_PI_2);

        // remove the first and last cone
        left_cones_.erase(left_cones_.begin());
        right_cones_.erase(right_cones_.begin());
        left_cones_.pop_back();
        right_cones_.pop_back();
    }
};

#endif // TRACK_GENERATOR_HPP_