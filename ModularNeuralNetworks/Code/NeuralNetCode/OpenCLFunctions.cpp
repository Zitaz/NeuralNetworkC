#include "OpenCLFunctions.h"

#include <Windows.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <assert.h>

namespace OpenCL
{
	Data Data::instace;
	Kernels Kernels::instace;

	void InitializeData()
	{
		//cl_int err;
		
		cl::Platform::get(&Data::instace.platform);
		
		std::vector<cl::Device> devices;
		Data::instace.platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

		for (unsigned i = 0; i < devices.size(); i++)
		{
			auto vendor = devices[i].getInfo<CL_DEVICE_VENDOR>();
			auto version = devices[i].getInfo<CL_DEVICE_VERSION>();
			auto max_work_item_size = devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
			auto local_mem_size = devices[i].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
			std::cout << vendor << std::endl << version << std::endl;

			for (unsigned c = 0; c < max_work_item_size.size(); c++)
			{
				std::cout << "Max work item size: " << max_work_item_size[c] << std::endl;
			}
			std::cout << "Local mem size: " << local_mem_size << std::endl;
		}
		
		assert(devices.size() != 0);
		Data::instace.device = devices.back();
		Data::instace.context = cl::Context(Data::instace.device);
		//Has to be in order queue
		Data::instace.queue = new cl::CommandQueue(Data::instace.context, Data::instace.device);
		Data::instace.program = OpenCL::CreateProgram("./OpenCLFunctions.cl");
	}

	void InitializeKernels()
	{
		cl_int error = 0;
		Kernels::instace.feedforward_fc_kernel[Net::Types::LINEAR] = cl::Kernel(Data::instace.program, "FeedForwardFCLinear", &error);//TODO: is there a better way to avoid branching?
		assert(error == 0);
		Kernels::instace.feedforward_fc_kernel[Net::Types::SIGMOID] = cl::Kernel(Data::instace.program, "FeedForwardFCSigmoid", &error);
		assert(error == 0);
		Kernels::instace.feedforward_fc_kernel[Net::Types::LEAKY_RELU] = cl::Kernel(Data::instace.program, "FeedForwardFCRelu", &error);
		assert(error == 0);

		Kernels::instace.backprop_fc_kernel[Net::Types::LEAKY_RELU] = cl::Kernel(Data::instace.program, "BackpropFCRelu", &error);
		assert(error == 0);

		Kernels::instace.feedforward_conv_kernel[Net::Types::LEAKY_RELU] = cl::Kernel(Data::instace.program, "FeedForwardConvRelu", &error);
		assert(error == 0);
		Kernels::instace.feedforward_conv_kernel[Net::Types::LINEAR] = cl::Kernel(Data::instace.program, "FeedForwardConvLinear", &error);
		assert(error == 0);

		Kernels::instace.calc_gradient_conv_kernel[Net::Types::LINEAR] = cl::Kernel(Data::instace.program, "CalcGradientConvLinear", &error);
		assert(error == 0);
		Kernels::instace.calc_gradient_conv_kernel[Net::Types::LEAKY_RELU] = cl::Kernel(Data::instace.program, "CalcGradientConvRelu", &error);
		assert(error == 0);

		Kernels::instace.update_weights_conv_kernel = cl::Kernel(Data::instace.program, "UpdateWeightsConv", &error);
		assert(error == 0);

		Kernels::instace.backprop_output_kernel[Net::Types::LINEAR] = cl::Kernel(Data::instace.program, "BackpropOutputLinear", &error);
		assert(error == 0);
		Kernels::instace.backprop_output_kernel[Net::Types::SIGMOID] = cl::Kernel(Data::instace.program, "BackpropOutputSigmoid", &error);
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

	cl::Program CreateProgram(const char* file_path)
	{
		std::vector<cl::Device> devices;
		devices.push_back(Data::instace.device);

		std::ifstream stream(file_path);
		std::string src(std::istreambuf_iterator<char>(stream), (std::istreambuf_iterator<char>()));

		cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

		cl::Program program(Data::instace.context, sources);

		cl_int err = program.build(devices);

		if (err)
		{
			//std::cout << err.what() << std::endl; works with cl::Error
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(Data::instace.device);
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(Data::instace.device);
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(Data::instace.device);
		}

		return program;
	}
}