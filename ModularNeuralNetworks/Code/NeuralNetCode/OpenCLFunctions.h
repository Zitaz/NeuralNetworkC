#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "NetTypes.h"

#include <CL/cl.h>
#include <CL/cl.hpp>

namespace OpenCL
{
	typedef struct Data
	{
		cl_platform_id platform;
		cl_device_id device;
		cl_context context;
		cl_program program;
		cl_command_queue queue;

		static Data instace;
	} Data;

	//Index = activation function
	typedef struct Kernels 
	{
		cl_kernel feedforward_fc_kernel[Net::Types::NUM_FUNCTIONS];
		cl_kernel backprop_fc_kernel[Net::Types::NUM_FUNCTIONS];
		cl_kernel feedforward_conv_kernel[Net::Types::NUM_FUNCTIONS];
		cl_kernel calc_gradient_conv_kernel[Net::Types::NUM_FUNCTIONS];
		cl_kernel update_weights_conv_kernel;
		cl_kernel backprop_output_kernel[Net::Types::NUM_FUNCTIONS];

		static Kernels instace;
	} Kernels;

	void InitializeData();

	void InitializeKernels();

	void DeinitializeData();

	//Pointless only one row
	cl::Kernel CreateKernel(const char* function_name, cl::Program program, cl_int& err);

	void CreateProgram(const char* file_path);
}