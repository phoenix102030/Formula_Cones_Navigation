#ifndef SPEED_PROFILER_HPP_
#define SPEED_PROFILER_HPP_

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>
#include <chrono>
#include <random>
#include <string>
#include "waypoint.hpp"


float triangleArea(Waypoint& wp1, Waypoint& wp2, Waypoint& wp3) {
    float x1 = wp1.x;
    float y1 = wp1.y;
    float x2 = wp2.x;
    float y2 = wp2.y;
    float x3 = wp3.x;
    float y3 = wp3.y;
    return 0.5 * abs(x1*y2 + x2*y3 + x3*y1 - x1*y3 - x2*y1 - x3*y2);
}

float radiusCurvature(Waypoint& wp1, Waypoint& wp2, Waypoint& wp3) {
    float area = triangleArea(wp1, wp2, wp3);
    float a = (wp1 - wp2).norm();
    float b = (wp2 - wp3).norm();
    float c = (wp3 - wp1).norm();
    return a * b * c / (4 * area + 1e-4);
}



/* 
 * Add speed profile to waypoints
 * 
 * @param waypoints: Ordered vector of waypoints from start to end
 * @param start_speed: start speed (or current speed)
 * @param end_speed: end speed (or target speed)
 * @param max_speed: maximum speed allowed
 * @param max_accel: maximum acceleration possible
 * @param max_decel: maximum deceleration possible
 * @param max_lat_accel: maximum lateral acceleration possible
*/
void addSpeedProfile(std::vector<Waypoint>& waypoints,
                     float start_speed = 0.0,
                     float end_speed = 15.0,
                     float max_speed = 20.0, 
                     float max_accel = 15.0,
                     float max_decel = -12.0,
                     float max_lat_accel = 12.0) {
    
    if (waypoints.size() < 2) {
        return;
    }
    
    waypoints[0].speed = start_speed;

    // Forward pass
    for (int i = 1; i < waypoints.size() - 1; i++) {
        float prev_speed = waypoints[i - 1].speed;

        int prev_idx = std::max(0, i - 1);
        int next_idx = std::min(static_cast<int>(waypoints.size()) - 1, i + 1);
        Waypoint wp1 = waypoints[prev_idx];
        Waypoint wp2 = waypoints[i];
        Waypoint wp3 = waypoints[next_idx];
        float radius = radiusCurvature(wp1, wp2, wp3);

        wp1 = waypoints[i - 1];
        wp2 = waypoints[i];
        wp3 = waypoints[i + 1];
        float dist = (wp3 - wp2).norm();

        float possible_speed = sqrt(prev_speed * prev_speed + 2 * max_accel * dist);
        float max_speed_curvature = sqrt(max_lat_accel * radius);

        float speed = std::min({max_speed, possible_speed, max_speed_curvature});

        waypoints[i].speed = speed;
    }

    waypoints[waypoints.size() - 1].speed = end_speed;

    // Backward pass
    for (int i = waypoints.size() - 2; i > 0; i--) {
        float next_speed = waypoints[i + 1].speed;
        Waypoint wp2 = waypoints[i];
        Waypoint wp3 = waypoints[i + 1];
        float dist = (wp3 - wp2).norm();
        float possible_speed = sqrt(next_speed * next_speed + 2 * max_decel * dist);
        float speed = std::min({possible_speed, max_speed, waypoints[i].speed});
        waypoints[i].speed = speed;
    }
}


#endif // SPEED_PROFILER_HPP_