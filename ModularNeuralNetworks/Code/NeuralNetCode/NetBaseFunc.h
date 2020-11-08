#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "NetUtility.h"
#include "CL/cl.hpp"
#include "NetComponents.h"

#include <vector>

namespace Net
{
	namespace BaseFunc
	{
		void SetValuesToZero(LayerData& layer);

		void SetValues(LayerData& layer, Types::LayerValues values);

		Types::LayerValues GetValues(LayerData& layer);

		void ActivateLayer(LayerData& layer);
	}
}