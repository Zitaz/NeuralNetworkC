#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "NetTypes.h"

#include <CL/cl.hpp>

namespace OpenCL
{
	struct Data
	{
		cl::Platform platform;
		cl::Device device;
		cl::Context context;
		cl::Program program;
		cl::CommandQueue* queue;

		static Data instace;
	};

	//Index = activation function
	struct Kernels 
	{
		cl::Kernel feedforward_fc_kernel[Net::Types::NUM_FUNCTIONS];
		cl::Kernel backprop_fc_kernel[Net::Types::NUM_FUNCTIONS];
		cl::Kernel feedforward_conv_kernel[Net::Types::NUM_FUNCTIONS];
		cl::Kernel calc_gradient_conv_kernel[Net::Types::NUM_FUNCTIONS];
		cl::Kernel update_weights_conv_kernel;
		cl::Kernel backprop_output_kernel[Net::Types::NUM_FUNCTIONS];

		static Kernels instace;
	};

	void InitializeData();

	void InitializeKernels();

	void DeinitializeData();

	//Pointless only one row
	cl::Kernel CreateKernel(const char* function_name, cl::Program program, cl_int& err);

	cl::Program CreateProgram(const char* file_path);
}