#ifndef WAYPOINT_HPP_
#define WAYPOINT_HPP_

#include "cone.hpp"

struct Waypoint {
    float x;
    float y;
    float speed;
    float track_width;

    Cone left_cone;
    Cone right_cone;

    Waypoint() {}

    Waypoint(float x, float y, float speed) {
        this->x = x;
        this->y = y;
        this->speed = speed;
    }

    Waypoint(float x, float y) {
        this->x = x;
        this->y = y;
        this->speed = 0;
    }

    Waypoint(float x, float y, Cone left_cone, Cone right_cone) {
        this->x = x;
        this->y = y;
        this->speed = speed;
        this->left_cone = left_cone;
        this->right_cone = right_cone;
        this->track_width = (left_cone - right_cone).norm();
    }

    Waypoint(float x, float y, float speed, Cone left_cone, Cone right_cone) {
        this->x = x;
        this->y = y;
        this->speed = speed;
        this->left_cone = left_cone;
        this->right_cone = right_cone;
        this->track_width = (left_cone - right_cone).norm();
    }

    Waypoint(const Waypoint &waypoint) {
        this->x = waypoint.x;
        this->y = waypoint.y;
        this->speed = waypoint.speed;
        this->left_cone = waypoint.left_cone;
        this->right_cone = waypoint.right_cone;
        this->track_width = waypoint.track_width;
    }

    Waypoint operator+(const Waypoint& waypoint) const {
        return Waypoint(x + waypoint.x, y + waypoint.y, left_cone + waypoint.left_cone, right_cone + waypoint.right_cone);
    }

    Waypoint operator-(const Waypoint& waypoint) const {
        return Waypoint(x - waypoint.x, y - waypoint.y, left_cone - waypoint.left_cone, right_cone - waypoint.right_cone);
    }

    Waypoint operator*(const float& scalar) const {
        return Waypoint(x * scalar, y * scalar);
    }

    Waypoint operator/(const float& scalar) const {
        return Waypoint(x / scalar, y / scalar);
    }

    void rotate(float angle) {
        float new_x = x * cos(angle) - y * sin(angle);
        float new_y = x * sin(angle) + y * cos(angle);
        x = new_x;
        y = new_y;
        left_cone.rotate(angle);
        right_cone.rotate(angle);
    }

    float norm() const {
        return sqrt(x * x + y * y);
    }
};


#endif // WAYPOINT_HPP_