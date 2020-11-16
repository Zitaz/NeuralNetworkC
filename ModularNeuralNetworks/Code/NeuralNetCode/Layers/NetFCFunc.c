#include "NetFCFunc.h"
#include "../NetBaseFunc.h"

#include <assert.h>

void Net_FCFeedForward(Net_LayerData* current_layer, Net_LayerData* next_layer, Net_CLData* cl_data)
{
	if (!current_layer->_use_open_CL) 
	{
		float* weights = current_layer->_weights_data->_weights;
		float* values = current_layer->_neurons_data->_values;
		float* values_next = next_layer->_neurons_data->_values;

		const unsigned next_num_neurons = next_layer->_neurons_data->_num_neurons;
		const unsigned current_num_neurons_with_bias = current_layer->_fC_layer_data->_num_neurons_with_bias;

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

		Net_BaseActivateLayer(next_layer);
	}
	else
	{
		//OpenCL error
		cl_int error = 0;

		//Get opencl kernel and _queue 
		cl_kernel kernel = cl_data->_kernals._feedforward_fc_kernel[next_layer->_function];
		cl_command_queue _queue = cl_data->_queue;

		//FeedForward
		error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &current_layer->_weights_CL_data->_buffer_weights);//TODO: Can we preprocess weights so we don't have to take so many cache misses on the gpu?
		assert(error == 0);
		error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &current_layer->_neurons_CL_data->_buffer_values);
		assert(error == 0);
		error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &next_layer->_neurons_CL_data->_buffer_values);
		assert(error == 0);
		error = clSetKernelArg(kernel, 3, sizeof(cl_int), &current_layer->_fC_layer_data->_num_neurons_with_bias);
		assert(error == 0);

		error = clFinish(_queue);
		assert(error == 0);

		size_t global_range[] = { next_layer->_neurons_CL_data->_num_neurons, 0, 0 };
		error = clEnqueueNDRangeKernel(_queue, kernel, 1, NULL, global_range, NULL, 0, NULL, NULL);
		assert(error == 0);

		error = clFinish(_queue);//Probably don't need this. Just using it for debugging
		assert(error == 0);
	}
}

void Net_FCBackprop(Net_LayerData* current_layer, Net_LayerData* next_layer, Net_CLData* cl_data, const float eta, const float momentum)
{
	if (!current_layer->_use_open_CL) {
		//Next_layer function?
		const Net_ActivationFunction function = current_layer->_function;
		const unsigned num_neurons_next = next_layer->_neurons_data->_num_neurons;
		const unsigned num_neurons_with_bias = current_layer->_fC_layer_data->_num_neurons_with_bias;

		const float* next_gradient = next_layer->_neurons_data->_gradient;
		const float* values = current_layer->_neurons_data->_values;

		float* gradient = current_layer->_neurons_data->_gradient;
		float* delta_weights = current_layer->_weights_data->_delta_weights;
		float* weights = current_layer->_weights_data->_weights;

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

			gradient[i] = target * Net_UtilGetDerivativ(function, values[i]);
		}
	}
	else
	{
		//OpenCL error
		cl_int error = 0;

		//Get opencl kernel and _queue 
		cl_kernel kernel = cl_data->_kernals._backprop_fc_kernel[current_layer->_function];
		cl_command_queue _queue = cl_data->_queue;

		//FeedForward
		error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &current_layer->_weights_CL_data->_buffer_weights);
		assert(error == 0);
		error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &current_layer->_weights_CL_data->_buffer_delta_weights);
		assert(error == 0);
		error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &current_layer->_neurons_CL_data->_buffer_gradient);
		assert(error == 0);
		error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &next_layer->_neurons_CL_data->_buffer_gradient);
		assert(error == 0);
		error = clSetKernelArg(kernel, 4, sizeof(cl_mem), &current_layer->_neurons_CL_data->_buffer_values);
		assert(error == 0);
		error = clSetKernelArg(kernel, 5, sizeof(cl_float), &eta);
		assert(error == 0);
		error = clSetKernelArg(kernel, 6, sizeof(cl_float), &momentum);
		assert(error == 0);
		error = clSetKernelArg(kernel, 7, sizeof(cl_int), &next_layer->_neurons_CL_data->_num_neurons);
		assert(error == 0);

		error = clFinish(_queue);
		assert(error == 0);

		size_t global_range[] = { current_layer->_fC_layer_data->_num_neurons_with_bias, 0, 0 };
		error = clEnqueueNDRangeKernel(_queue, kernel, 1, NULL, global_range, NULL, 0, NULL, NULL);
		assert(error == 0);

		error = clFinish(_queue);//Probobly dont need this. just using it for debuging
		assert(error == 0);
	}
}