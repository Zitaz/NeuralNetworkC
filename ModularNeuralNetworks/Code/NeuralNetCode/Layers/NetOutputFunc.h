#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "../NetBaseFunc.h"
#include "CL/cl.hpp"
#include "../NetComponents.h"

#include <vector>

namespace Net
{
	namespace OutputFunc 
	{
		void Backprop(LayerData& current_layer, const Types::LayerValues& target_output);
		float GetLoss(LayerData& current_layer, const Types::LayerValues& target_output);
	}
}