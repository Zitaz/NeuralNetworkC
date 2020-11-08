#include "NetFCFunc.h"
#include "../OpenCLFunctions.h"
#include "../NetBaseFunc.h"

//#include <chrono>
//#include <iostream>
#include <assert.h>


namespace Net
{
	namespace FCFunc
	{
		void FeedForward(LayerData& current_layer, LayerData& next_layer)
		{
			if (!current_layer._use_open_CL) 
			{
				float* weights = current_layer._weights_data->_weights;
				float* values = current_layer._neurons_data->_values;
				float* values_next = next_layer._neurons_data->_values;

				const unsigned next_num_neurons = next_layer._neurons_data->_num_neurons;
				const unsigned current_num_neurons_with_bias = current_layer._fC_layer_data->_num_neurons_with_bias;

				for (size_t next_index = 0; next_index < next_num_neurons; next_index++)
				{
					values_next[next_index] = 0.0f;
				}

				for (size_t current_index = 0; current_index < current_num_neurons_with_bias; current_index++)
				{
					for (size_t next_index = 0; next_index < next_num_neurons; next_index++)
					{
						float weight = weights[current_index * next_num_neurons + next_index];
						float value = values[current_index];
						values_next[next_index] += value * weight;
					}
				}

				Net::BaseFunc::ActivateLayer(next_layer);
			}
			else
			{
				//OpenCL error
				cl_int error = 0;

				//Get opencl kernel and queue 
				cl::Kernel& kernel = OpenCL::Kernels::instace.feedforward_fc_kernel[next_layer._function];
				cl::CommandQueue& queue = *OpenCL::Data::instace.queue;

				//FeedForward
				error = kernel.setArg(0, current_layer._weights_CL_data->_buffer_weights);//TODO: Can we preprocess weights so we don't have to take so many cache misses on the gpu?
				error = kernel.setArg(1, current_layer._neurons_CL_data->_buffer_values);
				error = kernel.setArg(2, next_layer._neurons_CL_data->_buffer_values);
				error = kernel.setArg(3, (cl_int)current_layer._fC_layer_data->_num_neurons_with_bias);

				error = queue.finish();
				error = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(next_layer._neurons_CL_data->_num_neurons));
				error = queue.finish();//Probably don't need this. Just using it for debugging
				assert(error == 0);
			}
		}

		void Backprop(LayerData& current_layer, LayerData& next_layer, const float eta, const float momentum)
		{
			if (!current_layer._use_open_CL) {
				//Next_layer function?
				const Types::ActivationFunction function = current_layer._function;
				const unsigned num_neurons_next = next_layer._neurons_data->_num_neurons;
				const unsigned num_neurons_with_bias = current_layer._fC_layer_data->_num_neurons_with_bias;

				const float* next_gradient = next_layer._neurons_data->_gradient;
				const float* values = current_layer._neurons_data->_values;

				float* gradient = current_layer._neurons_data->_gradient;
				float* delta_weights = current_layer._weights_data->_delta_weights;
				float* weights = current_layer._weights_data->_weights;

				for (size_t i = 0; i < num_neurons_with_bias; i++)
				{
					float target = 0.0f;

					for (size_t j = 0; j < num_neurons_next; j++)
					{
						{//Calc gradient
							target += weights[i * num_neurons_next + j] * next_gradient[j];
						}

						{//Update weight
							float new_delta_weight =
								eta
								* values[i]
								* next_gradient[j]
								+ momentum
								* delta_weights[i * num_neurons_next + j];

							delta_weights[i * num_neurons_next + j] = new_delta_weight;
							weights[i * num_neurons_next + j] += new_delta_weight;
						}
					}

					gradient[i] = target * Utility::GetDerivativ(function, values[i]);
				}
			}
			else
			{
				//OpenCL error
				cl_int error = 0;

				//Get opencl kernel and queue 
				cl::Kernel& kernel = OpenCL::Kernels::instace.backprop_fc_kernel[current_layer._function];
				cl::CommandQueue& queue = *OpenCL::Data::instace.queue;

				//FeedForward
				error = kernel.setArg(0, current_layer._weights_CL_data->_buffer_weights);
				error = kernel.setArg(1, current_layer._weights_CL_data->_buffer_delta_weights);
				error = kernel.setArg(2, current_layer._neurons_CL_data->_buffer_gradient);
				error = kernel.setArg(3, next_layer._neurons_CL_data->_buffer_gradient);
				error = kernel.setArg(4, current_layer._neurons_CL_data->_buffer_values);
				error = kernel.setArg(5, (cl_float)eta);
				error = kernel.setArg(6, (cl_float)momentum);
				error = kernel.setArg(7, (cl_int)next_layer._neurons_CL_data->_num_neurons);

				error = queue.finish();
				error = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(current_layer._fC_layer_data->_num_neurons_with_bias));
				error = queue.finish();//Probobly dont need this. just using it for debuging
				assert(error == 0);
			}
		}
	}
}