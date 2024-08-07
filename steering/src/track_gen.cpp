#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <thread>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <omp.h>

#include "common.hpp"
#include "track_generator.hpp"

namespace py = pybind11;

typedef std::pair<py::array_t<float>, py::array_t<float>> PyArrayPair;
typedef std::pair<py::array_t<float>, float> PyArrayFloatPair;



void fillTrackDataPoint(float* img_ptr, 
                        float& angle, 
                        float propagation_dist, 
                        float detection_prob, 
                        int max_false_positives, 
                        int img_size, 
                        float img_resolution,
                        float max_prop_angle) {

    Track track;
    track.buildRealisticFSTrack(detection_prob, max_false_positives);
    track.renormalizeTrack(propagation_dist, max_prop_angle);
    
    std::vector<Point> cones;
    track.getAllCones(cones);

    angle = track.getPropagationAngle(propagation_dist);

    for (int i = 0; i < cones.size(); i++) {
        float x_car = cones[i].x / img_resolution;
        float y_car = cones[i].y / img_resolution;
        int x_img = static_cast<int>(img_size / 2.0 - y_car);
        int y_img = static_cast<int>(img_size / 1.5 - x_car);

        if (x_img >= 0 && x_img < img_size && y_img >= 0 && y_img < img_size) {
            img_ptr[y_img * img_size + x_img] = 1.0f;
        }
    }
}


void conesToImg(float* img_ptr, 
                std::vector<Point>& cones, 
                int img_size, 
                float img_resolution) {

    for (int i = 0; i < cones.size(); i++) {
        float x_car = cones[i].x / img_resolution;
        float y_car = cones[i].y / img_resolution;
        int x_img = static_cast<int>(img_size / 2.0 - y_car);
        int y_img = static_cast<int>(img_size / 2.0 - x_car);

        if (x_img >= 0 && x_img < img_size && y_img >= 0 && y_img < img_size) {
            img_ptr[y_img * img_size + x_img] = 1.0f;
        }
    }
}


void genConeClassData(float* img_ptr, 
                      float* cone_class_ptr, 
                      float detection_prob, 
                      int max_false_positives, 
                      int img_size, 
                      float img_resolution) {

    Track track;
    track.buildRealisticFSTrack(1.1, max_false_positives);

    std::vector<Point> cones;
    track.getAllCones(cones);
    
    std::vector<Point> center_points;
    track.getCenterPoints(center_points);

    float left_or_right = random_uniform(-1.0, 1.0);

    std::vector<Point> side_cones;
    if (left_or_right < 0.0) {
        track.getLeftCones(side_cones);
        cone_class_ptr[0] = 1.0f;
    }
    else {
        track.getRightCones(side_cones);
        cone_class_ptr[1] = 1.0f;
    }

    int cone_idx = static_cast<int>(random_uniform(1.0f, static_cast<float>(side_cones.size() - 2)));

    int max_center_idx = center_points.size() - 1;
    int center_idx = cone_idx * track.getUseEvery();

    if (center_idx == max_center_idx) {
        center_idx = max_center_idx - 1;
    }

    Point side_cone = side_cones[cone_idx];
    Point center1 = center_points[center_idx];
    Point center2 = center_points[center_idx + 1];

    float direction = atan2(center2.y - center1.y, center2.x - center1.x);

    for (Point& cone : cones) {
        cone -= side_cone;
    }

    float angle_noise = random_uniform(-0.1f, 0.1f);
    rotatePoints(cones, -direction);

    float switch_to_fp = random_uniform(0.0f, 1.0f);
    if (switch_to_fp < 0.3) {
        float fp_x = random_uniform(-2.0f, 2.0f);
        float fp_y = random_uniform(-10.0f, 10.0f);
        Point fp(fp_x, fp_y);

        for (Point& cone : cones) {
            cone -= fp;
        }
        cone_class_ptr[0] = 0.0f;
        cone_class_ptr[1] = 0.0f;
        cone_class_ptr[2] = 1.0f;
    }

    // remove random cones
    std::vector<Point> new_cones;
    for (int i = 0; i < cones.size(); i++) {
        if (random_uniform(0.0f, 1.0f) < detection_prob) {
            new_cones.push_back(cones[i]);
        }
    }

    new_cones.push_back(Point(0.0f, 0.0f));

    conesToImg(img_ptr, new_cones, img_size, img_resolution);
}


PyArrayPair generateFSDataSet(const int num_tracks,
                              const float propagation_dist, 
                              const float detection_prob, 
                              const int max_false_positives,
                              const float img_range,
                              const float img_resolution,
                              const float max_prop_angle) {
 
    const int img_size = static_cast<int>(img_range / img_resolution);

    py::array_t<float> images({num_tracks, img_size, img_size});
    py::array_t<float> angles(num_tracks);

    std::fill(images.mutable_data(), images.mutable_data() + images.size(), 0.0f);

    float* angles_ptr = static_cast<float*>(angles.request().ptr);
    auto img = images.mutable_unchecked<3>();

    #pragma omp parallel for
    for (int i = 0; i < num_tracks; i++) {
        float* img_ptr = &img(i, 0, 0);
        fillTrackDataPoint(img_ptr, angles_ptr[i], propagation_dist, detection_prob, max_false_positives, img_size, img_resolution, max_prop_angle);
    }
    return std::make_pair(images, angles);
}


PyArrayPair generateConeClassDataset(const int num_cone_imgs,
                                     const float detection_prob, 
                                     const int max_false_positives,
                                     const float img_range,
                                     const float img_resolution) {
 
    const int img_size = static_cast<int>(img_range / img_resolution);

    py::array_t<float> images({num_cone_imgs, img_size, img_size});
    py::array_t<float> classes({num_cone_imgs, 3});

    std::fill(images.mutable_data(), images.mutable_data() + images.size(), 0.0f);
    std::fill(classes.mutable_data(), classes.mutable_data() + classes.size(), 0.0f);

    auto classes_ptr = classes.mutable_unchecked<2>();
    auto imgs = images.mutable_unchecked<3>();

    for (int i = 0; i < num_cone_imgs; i++) {
        float* img_ptr = &imgs(i, 0, 0);
        float* cone_class_ptr = &classes_ptr(i, 0);
        genConeClassData(img_ptr, cone_class_ptr, detection_prob, max_false_positives, img_size, img_resolution);
    }
    return std::make_pair(images, classes);
}


PyArrayFloatPair getTrackCones(const float propagation_dist, 
                               const float detection_prob, 
                               const int max_false_positives,
                               const float max_prop_angle) {
 
    Track track;
    track.buildRealisticFSTrack(detection_prob, max_false_positives);
    track.renormalizeTrack(propagation_dist, max_prop_angle);

    std::vector<Point> cones;
    track.getAllCones(cones);

    py::array_t<float> cones_arr({static_cast<int>(cones.size()), 2});
    auto cones_ptr = cones_arr.mutable_unchecked<2>();

    for (int i = 0; i < cones.size(); i++) {
        cones_ptr(i, 0) = cones[i].x;
        cones_ptr(i, 1) = cones[i].y;
    }

    return std::make_pair(cones_arr, track.getPropagationAngle(propagation_dist));
}


PYBIND11_MODULE(fsgenerator, m) {
    m.def("generate_fs_tracks", &generateFSDataSet, "Generates a track dataset",
          py::arg("num_tracks"), py::arg("propagation_dist"), py::arg("detection_prob"), 
          py::arg("max_false_positives"), py::arg("img_range"), py::arg("img_resolution"), 
          py::arg("max_prop_angle"));

    m.def("generate_cone_class_dataset", &generateConeClassDataset, "Generates a cone class dataset",
            py::arg("num_cone_imgs"), py::arg("detection_prob"), py::arg("max_false_positives"), 
            py::arg("img_range"), py::arg("img_resolution"));

    m.def("get_track_cones", &getTrackCones, "Gets the cones of a track",
            py::arg("propagation_dist"), py::arg("detection_prob"), py::arg("max_false_positives"), 
            py::arg("max_prop_angle"));
}