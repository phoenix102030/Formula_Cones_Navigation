
#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <vector>
#include <math.h>
#include <random>


unsigned SEED = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine GENERATOR(SEED);


template <typename T>
T random_uniform(T min, T max) {
    std::uniform_real_distribution<T> distribution(min, max);
    return distribution(GENERATOR);
}


template <typename T>
T random_normal(T mean, T std) {
    std::normal_distribution<T> distribution(mean, std);
    return distribution(GENERATOR);
}


struct Point {
    Point(float x, float y) {
        this->x = x;
        this->y = y;
    }

    Point() {
        this->x = 0;
        this->y = 0;
    }

    Point operator+(const Point& other) const {
        return Point(x + other.x, y + other.y);
    }

    Point operator-(const Point& other) const {
        return Point(x - other.x, y - other.y);
    }

    template <typename T>
    Point operator*(const T& other) const {
        return Point(x * other, y * other);
    }

    template <typename T>
    Point operator/(const T& other) const {
        return Point(x / other, y / other);
    }

    Point& operator+=(const Point& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    Point& operator-=(const Point& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    float norm() const {
        return sqrt(x * x + y * y);
    }

    void rotate(const float angle) {
        float c = cos(angle);
        float s = sin(angle);
        float x_new = x * c - y * s;
        float y_new = x * s + y * c;
        x = x_new;
        y = y_new;
    }

    Point rotated(const float angle) const {
        float c = cos(angle);
        float s = sin(angle);
        float x_new = x * c - y * s;
        float y_new = x * s + y * c;
        return Point(x_new, y_new);
    }

    float x;
    float y;
};


template <typename T>
Point operator*(const T& scalar, const Point& point) {
    return Point(point.x * scalar, point.y * scalar);
}

void rotatePoints(std::vector<Point>& points, const float angle) {
    float c = cos(angle);
    float s = sin(angle);
    for (int i = 0; i < points.size(); i++) {
        float x_new = points[i].x * c - points[i].y * s;
        float y_new = points[i].x * s + points[i].y * c;
        points[i].x = x_new;
        points[i].y = y_new;
    }
}


struct ImgPoint {

    ImgPoint(int x, int y) {
        this->x = x;
        this->y = y;
    }

    ImgPoint operator+(const ImgPoint& other) const {
        return ImgPoint(x + other.x, y + other.y);
    }

    ImgPoint operator-(const ImgPoint& other) const {
        return ImgPoint(x - other.x, y - other.y);
    }

    template <typename T>
    ImgPoint operator*(const T& other) const {
        return ImgPoint(x * other, y * other);
    }

    template <typename T>
    ImgPoint operator/(const T& other) const {
        return ImgPoint(x / other, y / other);
    }

    ImgPoint operator+=(const ImgPoint& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    ImgPoint operator-=(const ImgPoint& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    int x;
    int y;
};

#endif // COMMON_HPP_
