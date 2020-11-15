#include "OpenCLFunctions.h"

#include <Windows.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <assert.h>
#include <stdio.h>

namespace OpenCL
{
	Data Data::instace;
	Kernels Kernels::instace;

	void InitializeData()//TODO: Fix so you can pick correct device
	{
		const unsigned info_max_size = 500;
		cl_int err;

		//cl_uint num_platforms;
		//clGetPlatformIDs(NULL, NULL, &num_platforms);

		err = clGetPlatformIDs(1, &Data::instace.platform, NULL);//TODO: Get all platforms
		err = clGetDeviceIDs(Data::instace.platform, CL_DEVICE_TYPE_GPU, 1, &Data::instace.device, NULL);//TODO: Get all devices

		//Print info
		char info[info_max_size];

		err = clGetDeviceInfo(Data::instace.device, CL_DEVICE_VENDOR, info_max_size, info, NULL);
		printf("%s\n", info);

		err = clGetDeviceInfo(Data::instace.device, CL_DEVICE_VERSION, info_max_size, info, NULL);
		printf("%s\n", info);
		
		Data::instace.context = clCreateContext(NULL, 1, &Data::instace.device, NULL, NULL, NULL);
		Data::instace.queue = clCreateCommandQueue(Data::instace.context, Data::instace.device, NULL, NULL);

		//Has to be in order queue
		OpenCL::CreateProgram("./OpenCLFunctions.cl");

		InitializeKernels();
	}

	void InitializeKernels()
	{
		cl_int error = 0;
		Kernels::instace.feedforward_fc_kernel[Net::Types::LINEAR] = clCreateKernel(Data::instace.program, "FeedForwardFCLinear", &error);//TODO: is there a better way to avoid branching?
		assert(error == 0);
		Kernels::instace.feedforward_fc_kernel[Net::Types::SIGMOID] = clCreateKernel(Data::instace.program, "FeedForwardFCSigmoid", &error);
		assert(error == 0);
		Kernels::instace.feedforward_fc_kernel[Net::Types::LEAKY_RELU] = clCreateKernel(Data::instace.program, "FeedForwardFCRelu", &error);
		assert(error == 0);

		Kernels::instace.backprop_fc_kernel[Net::Types::LEAKY_RELU] = clCreateKernel(Data::instace.program, "BackpropFCRelu", &error);
		assert(error == 0);

		Kernels::instace.feedforward_conv_kernel[Net::Types::LEAKY_RELU] = clCreateKernel(Data::instace.program, "FeedForwardConvRelu", &error);
		assert(error == 0);
		Kernels::instace.feedforward_conv_kernel[Net::Types::LINEAR] = clCreateKernel(Data::instace.program, "FeedForwardConvLinear", &error);
		assert(error == 0);

		Kernels::instace.calc_gradient_conv_kernel[Net::Types::LINEAR] = clCreateKernel(Data::instace.program, "CalcGradientConvLinear", &error);
		assert(error == 0);
		Kernels::instace.calc_gradient_conv_kernel[Net::Types::LEAKY_RELU] = clCreateKernel(Data::instace.program, "CalcGradientConvRelu", &error);
		assert(error == 0);

		Kernels::instace.update_weights_conv_kernel = clCreateKernel(Data::instace.program, "UpdateWeightsConv", &error);
		assert(error == 0);

		Kernels::instace.backprop_output_kernel[Net::Types::LINEAR] = clCreateKernel(Data::instace.program, "BackpropOutputLinear", &error);
		assert(error == 0);
		Kernels::instace.backprop_output_kernel[Net::Types::SIGMOID] = clCreateKernel(Data::instace.program, "BackpropOutputSigmoid", &error);
		assert(error == 0);
	}

	void DeinitializeData()
	{
		delete Data::instace.queue;
	}

	//Pointless only one row
	cl::Kernel CreateKernel(const char* function_name, cl::Program program, cl_int& err)
	{
		return cl::Kernel(program, function_name);
	}

	void CreateProgram(const char* file_path)
	{
		cl_int error = 0;

		//Load file
		FILE* file;

		fopen_s(&file, file_path, "rb");

		assert(file != NULL);

		fseek(file, 0, SEEK_END);
		long size = ftell(file);
		rewind(file);

		char* buffer = (char*)malloc(sizeof(char) * size + 2);//+2 for \0
		buffer[size] = '\0';

		fread_s(buffer, sizeof(char) * size, sizeof(char) * size, 1, file);
		fclose(file);

		//Create program
		Data::instace.program = clCreateProgramWithSource(Data::instace.context, 1, (const char**)&buffer, NULL, &error);
		assert(error == 0);

		error = clBuildProgram(Data::instace.program, NULL, NULL, NULL, NULL, NULL);

		if (error)
		{
			error = clGetProgramBuildInfo(Data::instace.program, Data::instace.device, CL_PROGRAM_BUILD_STATUS, size, buffer, NULL);
			assert(error == 0);
			printf("Build Status: %s\n", buffer);

			error = clGetProgramBuildInfo(Data::instace.program, Data::instace.device, CL_PROGRAM_BUILD_OPTIONS, size, buffer, NULL);
			assert(error == 0);
			printf("Build Options:\t %s\n", buffer);

			error = clGetProgramBuildInfo(Data::instace.program, Data::instace.device, CL_PROGRAM_BUILD_LOG, size, buffer, NULL);
			assert(error == 0);
			printf("Build Log:\t %s\n", buffer);
		}

		free(buffer);
	}
}