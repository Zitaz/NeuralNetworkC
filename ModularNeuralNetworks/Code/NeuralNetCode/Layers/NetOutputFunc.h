#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "../NetBaseFunc.h"
#include "CL/cl.h"
#include "../NetComponents.h"
#include "../NetTypes.h"

extern "C" {
#include "../OpenCLFunctions.h"
}

namespace Net
{
	namespace OutputFunc 
	{
		void Backprop(LayerData& current_layer, const Net_ArrayF& target_output, Net_CLData* cl_data);
		float GetLoss(LayerData& current_layer, const Net_ArrayF& target_output, Net_CLData* cl_data);
	}
}