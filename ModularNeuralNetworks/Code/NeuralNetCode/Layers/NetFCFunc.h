#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "../NetComponents.h"
#include "../OpenCLFunctions.h"

void Net_FCFeedForward(Net_LayerData* current_layer, Net_LayerData* next_layer, Net_CLData* cl_data);
void Net_FCBackprop(Net_LayerData* current_layer, Net_LayerData* next_layer, Net_CLData* cl_data, const float eta, const float momentum);