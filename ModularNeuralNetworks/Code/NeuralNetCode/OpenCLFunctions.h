#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "NetTypes.h"

#include <CL/cl.h>

//Index = activation function
typedef struct Net_CLKernels
{
	cl_kernel _feedforward_fc_kernel[NET_ACTIVATION_FUNC_NUM_FUNCTIONS];
	cl_kernel _backprop_fc_kernel[NET_ACTIVATION_FUNC_NUM_FUNCTIONS];
	cl_kernel _feedforward_conv_kernel[NET_ACTIVATION_FUNC_NUM_FUNCTIONS];
	cl_kernel _calc_gradient_conv_kernel[NET_ACTIVATION_FUNC_NUM_FUNCTIONS];
	cl_kernel _update_weights_conv_kernel;
	cl_kernel _backprop_output_kernel[NET_ACTIVATION_FUNC_NUM_FUNCTIONS];

} Net_CLKernels;

typedef struct Net_CLData
{
	cl_platform_id _platform;
	cl_device_id _device;
	cl_context _context;
	cl_program _program;
	cl_command_queue _queue;

	Net_CLKernels _kernals;

} Net_CLData;

void Net_CLInitializeData(Net_CLData* data);
void Net_CLDeinitializeData(Net_CLData* data);