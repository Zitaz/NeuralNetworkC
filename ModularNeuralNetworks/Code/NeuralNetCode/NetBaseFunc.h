#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "NetUtility.h"
#include "CL/cl.h"
#include "NetComponents.h"

extern "C" {
#include "OpenCLFunctions.h"
}

#include <vector>

namespace Net
{
	namespace BaseFunc
	{
		void SetValuesToZero(LayerData& layer);

		void SetValues(LayerData& layer, Net_CLData* cl_data, Net_ArrayF values);

		Net_ArrayF GetValues(LayerData& layer, Net_CLData* cl_data);

		void ActivateLayer(LayerData& layer);
	}
}