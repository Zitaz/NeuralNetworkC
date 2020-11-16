#include "NetConvFunc.h"
#include "../NetUtility.h" 
#include "../NetBaseFunc.h"

#include <assert.h>

void Net_ConvFeedForward(Net_LayerData* current_layer, Net_LayerData* next_layer, Net_CLData* cl_data)
{
	if (!current_layer->_use_open_CL)
	{
		Net_ConvFeedForwardCpp(current_layer, next_layer);
	}
	else
	{
		Net_ConvFeedForwardCL(current_layer, next_layer, cl_data);
	}
}

void Net_ConvBackprop(Net_LayerData* current_layer, Net_LayerData* next_layer, Net_CLData* cl_data, const float eta, const float momentum)
{
	if (!current_layer->_use_open_CL)
	{
		Net_ConvCalcGradient(current_layer, next_layer);
		Net_ConvUpdateWeights(current_layer, next_layer, eta, momentum);
	}
	else
	{
		Net_ConvCalcGradientCL(current_layer, next_layer, cl_data);
		Net_ConvUpdateWeightsCL(current_layer, next_layer, cl_data, eta, momentum);
	}
}

void Net_ConvCalcGradient(Net_LayerData* current_layer, Net_LayerData* next_layer)
{
	const Net_ActivationFunction function = current_layer->_function;

	const unsigned length = current_layer->_conv_layer_data->_length;
	const unsigned length_pow = length * length;
	const unsigned depth = current_layer->_conv_layer_data->_depth;
	const unsigned stride = current_layer->_conv_layer_data->_stride;
	const unsigned filter_length = current_layer->_conv_layer_data->_filter_length;
	const unsigned filter_length_pow = filter_length * filter_length;
	const unsigned num_weights_filter = filter_length_pow * depth;
	const unsigned num_filters = current_layer->_conv_layer_data->_num_filters;
	const unsigned to_length = (length - filter_length) / stride + 1;
	const unsigned to_length_pow = to_length * to_length;
	const unsigned to_length_no_stride = (length - filter_length) + 1;

	const float* value = current_layer->_neurons_data->_values;
	const float* weight = current_layer->_weights_data->_weights;
	const float* next_gradient = next_layer->_neurons_data->_gradient;
	float* gradient = current_layer->_neurons_data->_gradient;

	for (unsigned weight_z = 0; weight_z < depth; ++weight_z)
	{
		const unsigned from_z_start_index = weight_z * length_pow;

		for (unsigned y = 0; y < length; y++)
		{
			const unsigned from_y_start_index = from_z_start_index + y * length;

			for (unsigned x = 0; x < length; x++)
			{
				const unsigned from_gradient_index = from_y_start_index + x;

				gradient[from_gradient_index] = 0.0f;

				for (unsigned filter_index = 0; filter_index < num_filters; ++filter_index)
				{
					//Calculating index of first weight for this filter
					const unsigned weight_start_index = filter_index * num_weights_filter;
					//Calculating weight z index in array
					const unsigned weight_z_index = weight_start_index + weight_z * filter_length_pow;
					//Calculating first to_value index for this filter
					const unsigned to_value_z_index = filter_index * to_length_pow;

					for (unsigned weight_y = 0; weight_y < filter_length; ++weight_y)
					{
						//position whitout stride. It needs to be devided with stride to get corect index in next layer
						int next_y = y - weight_y;

						{//Continue if there is no neuron at that position in next layer
							if ((next_y % stride) != 0 || next_y < 0 || next_y >= to_length_no_stride)
								continue;
						}

						const unsigned weight_y_index = weight_z_index + weight_y * filter_length;
						const unsigned to_value_y_index = to_value_z_index + next_y / stride * to_length;

						for (unsigned weight_x = 0; weight_x < filter_length; ++weight_x)
						{
							//position whitout stride. It needs to be devided  with stride to get corect index in next layer
							int next_x = x - weight_x;

							{//Continue if there is no neuron at that position in next layer
								if ((next_x % stride) != 0 || next_x < 0 || next_x >= to_length_no_stride)
									continue;
							}

							gradient[from_gradient_index] += weight[weight_y_index + weight_x] * next_gradient[to_value_y_index + next_x / stride];
						}
					}
				}

				gradient[from_gradient_index] = gradient[from_gradient_index] * Net_UtilGetDerivativ(function, value[from_gradient_index]);
			}
		}
	}
}

void Net_ConvUpdateWeights(Net_LayerData* current_layer, Net_LayerData* next_layer, const float eta, const float momentum)
{
	const unsigned length = current_layer->_conv_layer_data->_length;
	const unsigned length_pow = length * length;
	const unsigned num_neurons = current_layer->_neurons_data->_num_neurons;
	const unsigned depth = current_layer->_conv_layer_data->_depth;
	const unsigned stride = current_layer->_conv_layer_data->_stride;
	const unsigned filter_length = current_layer->_conv_layer_data->_filter_length;
	const unsigned filter_length_pow = filter_length * filter_length;
	const unsigned num_weights_filter = filter_length_pow * depth;
	const unsigned num_filters = current_layer->_conv_layer_data->_num_filters;
	const unsigned to_length = (length - filter_length) / stride + 1;
	const unsigned to_length_pow = to_length * to_length;

	const float* value_from_ptr = current_layer->_neurons_data->_values;
	const float* value_to_ptr = next_layer->_neurons_data->_values;
	const float* to_gradient_ptr = next_layer->_neurons_data->_gradient;
	float* weight_ptr = current_layer->_weights_data->_weights;
	float* delta_weight_ptr = current_layer->_weights_data->_delta_weights;

	for (unsigned filter_index = 0; filter_index < num_filters; ++filter_index)
	{
		for (unsigned weight_z = 0; weight_z < depth; ++weight_z)
		{
			for (unsigned weight_y = 0; weight_y < filter_length; ++weight_y)
			{
				for (unsigned weight_x = 0; weight_x < filter_length; ++weight_x)
				{
					float new_delta_weight = 0.0f;

					for (unsigned y = 0; y < to_length; y++)
					{
						for (unsigned x = 0; x < to_length; x++)
						{
							//Calculating first to_value index for this filter
							const unsigned to_value_z_index = filter_index * to_length_pow;
							//Calculating to_value index in array using to_z_start_index
							const unsigned to_value_y_index = to_value_z_index + y * to_length;

							const unsigned from_z_start_index = weight_z * length_pow;
							const unsigned from_y_start_index = from_z_start_index + ((y  * stride + weight_y) * length);

							new_delta_weight +=
								eta
								* value_from_ptr[from_y_start_index + weight_x + x * stride]
								* to_gradient_ptr[to_value_y_index + x];

						}
					}

					//Calculating index of first weight for this filter
					const unsigned weight_start_index = filter_index * num_weights_filter;
					//Calculating weight z index in array
					const unsigned weight_z_index = weight_start_index + weight_z * filter_length_pow;
					//Calculating weight z index in array
					const unsigned weight_y_index = weight_z_index + weight_y * filter_length;

					new_delta_weight = new_delta_weight
						+ momentum
						* delta_weight_ptr[weight_y_index + weight_x];

					delta_weight_ptr[weight_y_index + weight_x] = new_delta_weight;
					weight_ptr[weight_y_index + weight_x] += new_delta_weight;
				}
			}
		}
	}
}

void Net_ConvFeedForwardCpp(Net_LayerData* current_layer, Net_LayerData* next_layer)
{
	const unsigned length = current_layer->_conv_layer_data->_length;
	const unsigned length_pow = length * length;
	const unsigned depth = current_layer->_conv_layer_data->_depth;
	const unsigned filter_length = current_layer->_conv_layer_data->_filter_length;
	const unsigned filter_length_pow = filter_length * filter_length;
	const unsigned num_weights_filter = filter_length_pow * depth;
	const unsigned num_filters = current_layer->_conv_layer_data->_num_filters;
	const unsigned stride = current_layer->_conv_layer_data->_stride;
	const unsigned next_length = (length - filter_length) / stride + 1;
	const unsigned next_length_pow = next_length * next_length;

	const float* value_from_ptr = current_layer->_neurons_data->_values;
	const float* weight_ptr = current_layer->_weights_data->_weights;
	float* value_to_ptr = next_layer->_neurons_data->_values;

	for (size_t next_index = 0; next_index < next_layer->_neurons_data->_num_neurons; next_index++)
	{
		value_to_ptr[next_index] = 0.0f;
	}

	//If we use x and y to get index of from value, we will get the top left neuron in that kernal
	for (unsigned filter_index = 0; filter_index < num_filters; ++filter_index)
	{
		//Calculating first to_value index for this filter
		const unsigned to_value_z_index = filter_index * next_length_pow;
		//Calculating index of first weight for this filter
		const unsigned weight_start_index = filter_index * num_weights_filter;

		//It maybe more cash friendly to calculate weight_z here if you have many layers
		for (unsigned weight_z = 0; weight_z < depth; ++weight_z)
		{
			//Calculating weight z index in array
			const unsigned weight_z_index = weight_start_index + weight_z * filter_length_pow;
			//Calculating z start index in value from array
			const unsigned from_z_start_index = weight_z * length_pow;

			for (unsigned y = 0; y < next_length; y++)
			{
				//Calculating to_value index in array using to_z_start_index
				const unsigned to_value_y_index = to_value_z_index + y * next_length;

				for (unsigned x = 0; x < next_length; x++)
				{
					float value = 0;

					for (unsigned weight_y = 0; weight_y < filter_length; ++weight_y)
					{
						//Calculating weight z index in array
						const unsigned weight_y_index = weight_z_index + weight_y * filter_length;
						//Uses y to calculate y index in value from array
						//Don't need to add padding cus we whant to get the index in the top left corner of kernal
						//Then add weight_y to get to the corect neuron for that weight
						const unsigned from_y_start_index = from_z_start_index + ((y * stride + weight_y) * length);

						for (unsigned weight_x = 0; weight_x < filter_length; ++weight_x)
						{
							value += weight_ptr[weight_y_index + weight_x] * value_from_ptr[from_y_start_index + weight_x + x * stride];
						}
					}

					value_to_ptr[to_value_y_index + x] += value;
				}
			}
		}
	}

	Net_BaseActivateLayer(next_layer);//TODO: Maybe can do this in the loop?
}

void Net_ConvFeedForwardCL(Net_LayerData* current_layer, Net_LayerData* next_layer, Net_CLData* cl_data)
{
	const unsigned next_length = (current_layer->_conv_layer_data->_length - current_layer->_conv_layer_data->_filter_length) / current_layer->_conv_layer_data->_stride + 1;

	//OpenCL error
	cl_int error = 0;

	//Get opencl kernel and _queue 
	cl_kernel kernel = cl_data->_kernals._feedforward_conv_kernel[next_layer->_function];
	cl_command_queue _queue = cl_data->_queue;
	
	//FeedForward
	error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &current_layer->_weights_CL_data->_buffer_weights);
	assert(error == 0);
	error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &current_layer->_neurons_CL_data->_buffer_values);
	assert(error == 0);
	error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &next_layer->_neurons_CL_data->_buffer_values);
	assert(error == 0);
	error = clSetKernelArg(kernel, 3, sizeof(cl_int), &current_layer->_conv_layer_data->_stride);
	assert(error == 0);
	error = clSetKernelArg(kernel, 4, sizeof(cl_int), &current_layer->_conv_layer_data->_filter_length);
	assert(error == 0);
	error = clSetKernelArg(kernel, 5, sizeof(cl_int), &current_layer->_conv_layer_data->_depth);
	assert(error == 0);

	error = clFinish(_queue);
	assert(error == 0);

	size_t global_range[] = { next_length, next_length, current_layer->_conv_layer_data->_num_filters };
	error = clEnqueueNDRangeKernel(_queue, kernel, 3, NULL, global_range, NULL, 0, NULL, NULL);
	assert(error == 0);

	error = clFinish(_queue);
	assert(error == 0);
}

void Net_ConvUpdateWeightsCL(Net_LayerData* current_layer, Net_LayerData* next_layer, Net_CLData* cl_data, const float eta, const float momentum)
{
	const unsigned next_length = (current_layer->_conv_layer_data->_length - current_layer->_conv_layer_data->_filter_length) / current_layer->_conv_layer_data->_stride + 1;

	//OpenCL error
	cl_int error = 0;

	//Get opencl kernel and _queue 
	cl_kernel kernel = cl_data->_kernals._update_weights_conv_kernel;
	cl_command_queue _queue = cl_data->_queue;
	
	//FeedForward
	error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &current_layer->_weights_CL_data->_buffer_weights);
	assert(error == 0);
	error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &current_layer->_weights_CL_data->_buffer_delta_weights);
	assert(error == 0);
	error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &current_layer->_neurons_CL_data->_buffer_values);
	assert(error == 0);
	error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &next_layer->_neurons_CL_data->_buffer_gradient);
	assert(error == 0);
	error = clSetKernelArg(kernel, 4, sizeof(cl_int), &current_layer->_conv_layer_data->_stride);
	assert(error == 0);
	error = clSetKernelArg(kernel, 5, sizeof(cl_int), &current_layer->_conv_layer_data->_length);
	assert(error == 0);
	error = clSetKernelArg(kernel, 6, sizeof(cl_int), &current_layer->_conv_layer_data->_filter_length);
	assert(error == 0);
	error = clSetKernelArg(kernel, 7, sizeof(cl_float), &eta);
	assert(error == 0);
	error = clSetKernelArg(kernel, 8, sizeof(cl_float), &momentum);
	assert(error == 0);

	error = clFinish(_queue);
	assert(error == 0);

	size_t global_range[] = { current_layer->_conv_layer_data->_filter_length * current_layer->_conv_layer_data->_filter_length, current_layer->_conv_layer_data->_depth, current_layer->_conv_layer_data->_num_filters };
	error = clEnqueueNDRangeKernel(_queue, kernel, 3, NULL, global_range, NULL, 0, NULL, NULL);
	assert(error == 0);

	error = clFinish(_queue);
	assert(error == 0);
}

void Net_ConvCalcGradientCL(Net_LayerData* current_layer, Net_LayerData* next_layer, Net_CLData* cl_data)
{
	//OpenCL error
	cl_int error = 0;

	//Get opencl kernel and _queue 
	cl_kernel kernel = cl_data->_kernals._calc_gradient_conv_kernel[current_layer->_function];
	cl_command_queue queue = cl_data->_queue;

	//FeedForward
	error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &current_layer->_weights_CL_data->_buffer_weights);
	assert(error == 0);
	error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &current_layer->_neurons_CL_data->_buffer_values);
	assert(error == 0);
	error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &current_layer->_neurons_CL_data->_buffer_gradient);
	assert(error == 0);
	error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &next_layer->_neurons_CL_data->_buffer_gradient);
	assert(error == 0);
	error = clSetKernelArg(kernel, 4, sizeof(cl_int), &current_layer->_conv_layer_data->_stride);
	assert(error == 0);
	error = clSetKernelArg(kernel, 5, sizeof(cl_int), &current_layer->_conv_layer_data->_filter_length);
	assert(error == 0);
	error = clSetKernelArg(kernel, 6, sizeof(cl_int), &current_layer->_conv_layer_data->_num_filters);
	assert(error == 0);

	error = clFinish(queue);
	assert(error == 0);

	size_t global_range[] = { current_layer->_conv_layer_data->_length, current_layer->_conv_layer_data->_length, current_layer->_conv_layer_data->_depth };
	error = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_range, NULL, 0, NULL, NULL);
	assert(error == 0);

	error = clFinish(queue);
	assert(error == 0);
}