#ifndef CONE_HPP_
#define CONE_HPP_

#include <cmath>

struct Cone {
    float x = 0;
    float y = 0;
    float color_score = 0;
    uint8_t color = 0;

    Cone() {}

    Cone(float x, float y) {
        this->x = x;
        this->y = y;
    }

    Cone(float x, float y, float color_score) {
        this->x = x;
        this->y = y;
        this->color_score = color_score;
    }

    Cone(float x, float y, float color_score, uint8_t color) {
        this->x = x;
        this->y = y;
        this->color_score = color_score;
        this->color = color;
    }

    void rotate(float angle) {
        float new_x = x * cos(angle) - y * sin(angle);
        float new_y = x * sin(angle) + y * cos(angle);
        x = new_x;
        y = new_y;
    }

    float norm() const {
        return sqrt(x * x + y * y);
    }
};

#endif // CONE_HPP_