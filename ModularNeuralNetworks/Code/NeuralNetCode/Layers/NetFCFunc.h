#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "../NeuralNet.h"
#include "CL/cl.hpp"
#include "../NetComponents.h"

#include <vector>

namespace Net
{
	namespace FCFunc
	{
		void FeedForward(LayerData& current_layer, LayerData& next_layer, Net_CLData* cl_data);
		void Backprop(LayerData& current_layer, LayerData& next_layer, Net_CLData* cl_data, const float eta, const float momentum);
	}
}