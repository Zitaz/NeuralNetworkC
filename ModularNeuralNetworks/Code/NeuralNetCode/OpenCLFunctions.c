#include "OpenCLFunctions.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

void Net_CLInitializeData(Net_CLData* data)//TODO: Fix so you can pick correct _device
{
	cl_int error = 0;

	{//Set up  Net_CLData
		const unsigned info_max_size = 500;
		//cl_uint num_platforms;
		//clGetPlatformIDs(NULL, NULL, &num_platforms);

		error = clGetPlatformIDs(1, &data->_platform, NULL);//TODO: Get all platforms
		error = clGetDeviceIDs(data->_platform, CL_DEVICE_TYPE_GPU, 1, &data->_device, NULL);//TODO: Get all devices

		//Print info
		char info[500];

		error = clGetDeviceInfo(data->_device, CL_DEVICE_VENDOR, info_max_size, info, NULL);
		assert(error == 0);
		printf("%s\n", info);

		error = clGetDeviceInfo(data->_device, CL_DEVICE_VERSION, info_max_size, info, NULL);
		assert(error == 0);
		printf("%s\n", info);

		data->_context = clCreateContext(NULL, 1, &data->_device, NULL, NULL, NULL);
		data->_queue = clCreateCommandQueue(data->_context, data->_device, NULL, NULL);

		//Has to be in order _queue
	}

	{//Create program

		//Load file
		FILE* file;

		fopen_s(&file, "./OpenCLFunctions.cl", "rb");

		assert(file != NULL);

		fseek(file, 0, SEEK_END);
		long size = ftell(file);
		rewind(file);

		char* buffer = (char*)malloc(sizeof(char) * size + 2);//+2 for \0
		buffer[size] = '\0';

		fread_s(buffer, sizeof(char) * size, sizeof(char) * size, 1, file);
		fclose(file);

		//Create _program
		data->_program = clCreateProgramWithSource(data->_context, 1, (const char**)&buffer, NULL, &error);
		assert(error == 0);

		error = clBuildProgram(data->_program, NULL, NULL, NULL, NULL, NULL);

		if (error)
		{
			error = clGetProgramBuildInfo(data->_program, data->_device, CL_PROGRAM_BUILD_STATUS, size, buffer, NULL);
			assert(error == 0);
			printf("Build Status: %s\n", buffer);

			error = clGetProgramBuildInfo(data->_program, data->_device, CL_PROGRAM_BUILD_OPTIONS, size, buffer, NULL);
			assert(error == 0);
			printf("Build Options:\t %s\n", buffer);

			error = clGetProgramBuildInfo(data->_program, data->_device, CL_PROGRAM_BUILD_LOG, size, buffer, NULL);
			assert(error == 0);
			printf("Build Log:\t %s\n", buffer);
		}

		free(buffer);
	}

	{//Create kernals
		data->_kernals._feedforward_fc_kernel[NET_ACTIVATION_FUNC_LINEAR] = clCreateKernel(data->_program, "FeedForwardFCLinear", &error);//TODO: is there a better way to avoid branching?
		assert(error == 0);
		data->_kernals._feedforward_fc_kernel[NET_ACTIVATION_FUNC_SIGMOID] = clCreateKernel(data->_program, "FeedForwardFCSigmoid", &error);
		assert(error == 0);
		data->_kernals._feedforward_fc_kernel[NET_ACTIVATION_FUNC_LEAKY_RELU] = clCreateKernel(data->_program, "FeedForwardFCRelu", &error);
		assert(error == 0);

		data->_kernals._backprop_fc_kernel[NET_ACTIVATION_FUNC_LEAKY_RELU] = clCreateKernel(data->_program, "BackpropFCRelu", &error);
		assert(error == 0);

		data->_kernals._feedforward_conv_kernel[NET_ACTIVATION_FUNC_LEAKY_RELU] = clCreateKernel(data->_program, "FeedForwardConvRelu", &error);
		assert(error == 0);
		data->_kernals._feedforward_conv_kernel[NET_ACTIVATION_FUNC_LINEAR] = clCreateKernel(data->_program, "FeedForwardConvLinear", &error);
		assert(error == 0);

		data->_kernals._calc_gradient_conv_kernel[NET_ACTIVATION_FUNC_LINEAR] = clCreateKernel(data->_program, "CalcGradientConvLinear", &error);
		assert(error == 0);
		data->_kernals._calc_gradient_conv_kernel[NET_ACTIVATION_FUNC_LEAKY_RELU] = clCreateKernel(data->_program, "CalcGradientConvRelu", &error);
		assert(error == 0);

		data->_kernals._update_weights_conv_kernel = clCreateKernel(data->_program, "UpdateWeightsConv", &error);
		assert(error == 0);

		data->_kernals._backprop_output_kernel[NET_ACTIVATION_FUNC_LINEAR] = clCreateKernel(data->_program, "BackpropOutputLinear", &error);
		assert(error == 0);
		data->_kernals._backprop_output_kernel[NET_ACTIVATION_FUNC_SIGMOID] = clCreateKernel(data->_program, "BackpropOutputSigmoid", &error);
		assert(error == 0);
	}
}

void Net_CLDeinitializeData(Net_CLData* data)
{
	//delete Data::instace._queue;
}