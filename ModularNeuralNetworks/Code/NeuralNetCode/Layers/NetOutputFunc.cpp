#include "NetOutputFunc.h"

#include <assert.h>

namespace Net
{
	namespace OutputFunc
	{
		void Backprop(LayerData& current_layer, const Net_ArrayF & target_output, Net_CLData* cl_data)
		{
			if (!current_layer._use_open_CL)
			{
				unsigned num_neurons = current_layer._neurons_data->_num_neurons;

				float* values = current_layer._neurons_data->_values;
				float* gradient = current_layer._neurons_data->_gradient;

				assert(target_output._size == num_neurons);

				//Calculate gradients
				for (size_t i = 0; i < num_neurons; i++)
				{
					float value = values[i];

					float new_gradient = (target_output._data[i] - value) * Utility::GetDerivativ(current_layer._function, value);

					gradient[i] = new_gradient;
				}
			}
			else
			{
				assert(target_output._size == current_layer._neurons_CL_data->_num_neurons);

				//OpenCL error
				cl_int error = 0;

				//Get opencl kernel and _queue 
				cl_kernel& kernel = cl_data->_kernals._backprop_output_kernel[current_layer._function];
				cl_command_queue& _queue = cl_data->_queue;

				error = clEnqueueWriteBuffer(cl_data->_queue, current_layer._output_layer_CL_data->_target_value, CL_FALSE, 0, sizeof(float) * target_output._size, target_output._data, NULL, NULL, NULL);
				assert(error == 0);

				//FeedForward
				error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &current_layer._neurons_CL_data->_buffer_gradient);
				assert(error == 0);
				error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &current_layer._neurons_CL_data->_buffer_values);
				assert(error == 0);
				error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &current_layer._output_layer_CL_data->_target_value);
				assert(error == 0);
				
				error = clFinish(_queue);
				assert(error == 0);

				size_t global_range[] = { current_layer._neurons_CL_data->_num_neurons, 0, 0 };
				error = clEnqueueNDRangeKernel(_queue, kernel, 1, NULL, global_range, NULL, 0, NULL, NULL);
				assert(error == 0);

				error = clFinish(_queue);//Probobly dont's need this. just using it for debuging
				assert(error == 0);
			}
		}

		float GetLoss(LayerData& current_layer, const Net_ArrayF& target_output, Net_CLData* cl_data)
		{
			if (!current_layer._use_open_CL)
			{
				unsigned num_neurons = current_layer._neurons_data->_num_neurons;

				float* values = current_layer._neurons_data->_values;

				assert(target_output._size == num_neurons);

				float loss = 0.0f;

				//Calculate gradients
				for (size_t i = 0; i < num_neurons; i++)
				{
					loss += (target_output._data[i] - values[i]) * (target_output._data[i] - values[i]);
				}

				return sqrt(loss);
			}
			else
			{
				unsigned num_neurons = current_layer._neurons_CL_data->_num_neurons;

				Net_ArrayF values = BaseFunc::GetValues(current_layer, cl_data);

				assert(target_output._size == num_neurons);

				float loss = 0.0f;

				//Calculate gradients
				for (size_t i = 0; i < num_neurons; i++)
				{
					loss += (target_output._data[i] - values._data[i]) * (target_output._data[i] - values._data[i]);
				}

				Net_FreeArrayF(&values);

				return sqrt(loss);
			}
		}
	}
}