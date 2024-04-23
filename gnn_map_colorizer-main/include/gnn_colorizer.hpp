
#ifndef GNN_COLORIZER_HPP_
#define GNN_COLORIZER_HPP_

#include <torch/script.h>
#include <iostream>
#include <chrono>
#include <math.h>
#include <string>
#include <vector>

#include "cone.hpp"


class GNNColorizer {
public:
    GNNColorizer() {}

    GNNColorizer(std::string model_path,
                 float propagation_distance,
                 float max_prop_angle,
                 float max_input_range,
                 float max_track_width,
                 float association_threshold){

        prop_dist_ = propagation_distance;   
        max_prop_angle_ = max_prop_angle;
        max_input_range_ = max_input_range;
        max_track_width_ = max_track_width;
        association_threshold_ = association_threshold;

        // Load the model
        try {
            module = torch::jit::load(model_path);
        }
        catch (const c10::Error& e) {
            std::cerr << "Error loading the model" << std::endl;
        }
        module.eval();

        // Warmup
        int num_nodes = 50;
        torch::Tensor node_features = torch::rand({num_nodes, 2});
        for (int i = 0; i < 1000; i++) {
            auto output = module.forward({node_features}).toTensor().item<float>();
        }
    }

    void colorize(std::vector<Cone>& cone_map, float ego_x, float ego_y, float ego_yaw, int max_steps) {
        givePreviousColors(cone_map);

        std::vector<Cone> transformed_cones = cone_map;
        for (Cone& cone : transformed_cones) {
            cone.x -= ego_x;
            cone.y -= ego_y;
            cone.rotate(-ego_yaw);
        }

        for (int step = 0; step < max_steps; step++) {
            std::vector<Cone> cones_in_range = getConesInRange(transformed_cones);
            if (stopCriteria(cones_in_range)) { break; }

            torch::Tensor graph_input = conesToTorchGraph(cones_in_range);
            float prop_angle = GNNforward(graph_input) * max_prop_angle_;

            float local_wp_x = prop_dist_ * cos(prop_angle);
            float local_wp_y = prop_dist_ * sin(prop_angle);

            for (int i = 0; i < cone_map.size(); i++) {
                Cone& trans_cone = transformed_cones[i];
                Cone& global_cone = cone_map[i];

                trans_cone.x -= local_wp_x;
                trans_cone.y -= local_wp_y;
                trans_cone.rotate(-prop_angle);

                if (trans_cone.norm() < max_track_width_ / 1.5) {
                    global_cone.color_score += trans_cone.y / (prop_dist_ * (step + 1));
                }
            }
        }
        full_map_ = cone_map;
    }

private:
    float prop_dist_;
    float max_prop_angle_;
    float max_input_range_;
    float max_track_width_;
    float association_threshold_;

    torch::jit::Module module;

    std::vector<Cone> full_map_;

    void givePreviousColors(std::vector<Cone>& cones) {
        for (Cone& cone : cones) {
            int closest_index = findClosestCone(cone, full_map_);
            if (closest_index != -1) {
                cone.color_score += full_map_[closest_index].color_score;
            }
        }
    }

    int findClosestCone(const Cone& cone, const std::vector<Cone>& cones) {
        float min_dist = association_threshold_;
        int min_index = -1;
        for (int i = 0; i < cones.size(); i++) {
            float dist = sqrt(pow(cone.x - cones[i].x, 2) + pow(cone.y - cones[i].y, 2));
            if (dist < min_dist) {
                min_dist = dist;
                min_index = i;
            }
        }
        return min_index;
    }

    torch::Tensor conesToTorchGraph(const std::vector<Cone>& cones) {
        int num_nodes = cones.size();
        std::vector<float> data;
        data.reserve(num_nodes * 2);
        for (int i = 0; i < num_nodes; i++) {
            data.push_back(cones[i].x);
            data.push_back(cones[i].y);
        }

        torch::Tensor model_input = torch::from_blob(data.data(), {num_nodes, 2}, torch::kFloat);
        model_input = model_input.clone();
        return model_input;
    }

    std::vector<Cone> getConesInRange(const std::vector<Cone>& cones) {
        std::vector<Cone> cones_in_range;
        for (const Cone& cone : cones) {
            if (cone.norm() < max_input_range_) {
                cones_in_range.push_back(cone);
            }
        }
        return cones_in_range;
    }

    bool stopCriteria(const std::vector<Cone>& cones) {
        for (const Cone& cone : cones) {
            if (cone.x > 0) {
                return false;
            }
        }
        return true;
    }

    float GNNforward(const torch::Tensor& node_features) {
        return module.forward({node_features}).toTensor().item<float>();
    }
};

#endif // GNN_COLORIZER_HPP_

