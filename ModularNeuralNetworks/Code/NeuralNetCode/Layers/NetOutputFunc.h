#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "../NetBaseFunc.h"
#include "../OpenCLFunctions.h"
#include "../NetComponents.h"
#include "../NetUtility.h"

void Net_OutBackprop(Net_LayerData* current_layer, const Net_ArrayF* target_output, Net_CLData* cl_data);
float Net_OutGetLoss(Net_LayerData* current_layer, const Net_ArrayF* target_output, Net_CLData* cl_data);