#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "CL/cl.h"

#include "NetUtility.h"
#include "NetComponents.h"
#include "OpenCLFunctions.h"
#include "NetTypes.h"

void Net_BaseSetValuesToZero(Net_LayerData* layer);

void Net_BaseSetValues(Net_LayerData* layer, Net_CLData* cl_data, Net_ArrayF values);

Net_ArrayF Net_BaseGetValues(Net_LayerData* layer, Net_CLData* cl_data);

void Net_BaseActivateLayer(Net_LayerData* layer);