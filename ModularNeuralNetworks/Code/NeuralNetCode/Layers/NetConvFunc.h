#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "../NetComponents.h"
#include "../OpenCLFunctions.h"

void Net_ConvFeedForward(Net_LayerData* current_layer, Net_LayerData* next_layer, Net_CLData* cl_data);
void Net_ConvBackprop(Net_LayerData* current_layer, Net_LayerData* next_layer, Net_CLData* cl_data, const float eta, const float momentum);
static void Net_ConvCalcGradient(Net_LayerData* current_layer, Net_LayerData* next_layer);//TODO: Can we add CalcGradient to UpdateWeights on cpu?
static void Net_ConvUpdateWeights(Net_LayerData* current_layer, Net_LayerData* next_layer, const float eta, const float momentum);

static void Net_ConvFeedForwardCpp(Net_LayerData* current_layer, Net_LayerData* next_layer);
static void Net_ConvFeedForwardCL(Net_LayerData* current_layer, Net_LayerData* next_layer, Net_CLData* cl_data);
static void Net_ConvUpdateWeightsCL(Net_LayerData* current_layer, Net_LayerData* next_layer, Net_CLData* cl_data, const float eta, const float momentum);
static void Net_ConvCalcGradientCL(Net_LayerData* current_layer, Net_LayerData* next_layer, Net_CLData* cl_data);
