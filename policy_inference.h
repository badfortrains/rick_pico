#ifndef RICK_POLICY_INFERENCE_H
#define RICK_POLICY_INFERENCE_H

#include <algorithm>
#include <cmath>

#include "policy_weights.h"

namespace rick {

inline float swish(float x) {
    if (x >= 0.0f) return x / (1.0f + std::exp(-x));
    const float exp_x = std::exp(x);
    return x * exp_x / (1.0f + exp_x);
}

inline void dense_layer(
    const float *input, const float *weights, const float *biases,
    float *output, int in_features, int out_features, bool apply_swish) {
    std::copy_n(biases, out_features, output);
    // Flax kernels are [input, output]. Stream each row from flash once,
    // instead of striding across a whole matrix for each output neuron.
    // Each output retains the original input accumulation order.
    for (int input_index = 0; input_index < in_features; ++input_index) {
        const float value = input[input_index];
        const float *row = weights + input_index * out_features;
        for (int output_index = 0; output_index < out_features; ++output_index) {
            output[output_index] += value * row[output_index];
        }
    }
    if (apply_swish) {
        for (int index = 0; index < out_features; ++index) {
            output[index] = swish(output[index]);
        }
    }
}

class PolicyNetwork {
public:
    void infer(const float *raw_obs, float *action) {
        for (int index = 0; index < POLICY_OBS_DIM; ++index) {
            normalized_obs_[index] =
                (raw_obs[index] - POLICY_OBS_MEAN[index]) / POLICY_OBS_STD[index];
        }
        const float *input = normalized_obs_;
        float *output = buffer_a_;
        for (int layer = 0; layer < POLICY_DENSE_LAYER_COUNT; ++layer) {
            dense_layer(
                input, POLICY_LAYER_KERNELS[layer], POLICY_LAYER_BIASES[layer],
                output, POLICY_LAYER_IN_DIMS[layer], POLICY_LAYER_OUT_DIMS[layer],
                layer + 1 != POLICY_DENSE_LAYER_COUNT);
            input = output;
            output = output == buffer_a_ ? buffer_b_ : buffer_a_;
        }
        // Brax deterministic tanh-normal mode is tanh(first half of logits).
        for (int index = 0; index < POLICY_ACTION_DIM; ++index) {
            action[index] = std::tanh(input[index]);
        }
    }

private:
    float normalized_obs_[POLICY_OBS_DIM];
    float buffer_a_[POLICY_MAX_LAYER_DIM];
    float buffer_b_[POLICY_MAX_LAYER_DIM];
};

}  // namespace rick
#endif
