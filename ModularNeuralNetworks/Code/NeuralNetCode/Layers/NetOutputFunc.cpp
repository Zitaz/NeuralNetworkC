#include "NetOutputFunc.h"
#include "../OpenCLFunctions.h"

#include <assert.h>


namespace Net
{
	namespace OutputFunc
	{
		void Backprop(LayerData& current_layer, const Types::LayerValues & target_output)
		{
			if (!current_layer._use_open_CL)
			{
				unsigned num_neurons = current_layer._neurons_data->_num_neurons;

				float* values = current_layer._neurons_data->_values;
				float* gradient = current_layer._neurons_data->_gradient;

				assert(target_output.size() == num_neurons);

				//Calculate gradients
				for (size_t i = 0; i < num_neurons; i++)
				{
					float value = values[i];

					float new_gradient = (target_output[i] - value) * Utility::GetDerivativ(current_layer._function, value);

					gradient[i] = new_gradient;
				}
			}
			else
			{
				assert(target_output.size() == current_layer._neurons_CL_data->_num_neurons);

				//OpenCL error
				cl_int error = 0;

				//Get opencl kernel and queue 
				cl_kernel& kernel = OpenCL::Kernels::instace.backprop_output_kernel[current_layer._function];
				cl_command_queue& queue = OpenCL::Data::instace.queue;

				clEnqueueWriteBuffer(OpenCL::Data::instace.queue, current_layer._output_layer_CL_data->_target_value, CL_FALSE, 0, sizeof(float) * target_output.size(), &target_output[0], NULL, NULL, NULL);

				//FeedForward
				error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &current_layer._neurons_CL_data->_buffer_gradient);
				error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &current_layer._neurons_CL_data->_buffer_values);
				error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &current_layer._output_layer_CL_data->_target_value);
				
				clFinish(queue);
				size_t global_range[] = { current_layer._neurons_CL_data->_num_neurons, 0, 0 };
				error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_range, NULL, 0, NULL, NULL);
				clFinish(queue);//Probobly dont's need this. just using it for debuging
				assert(error == 0);
			}
		}

		float GetLoss(LayerData& current_layer, const Types::LayerValues& target_output)
		{
			if (!current_layer._use_open_CL)
			{
				unsigned num_neurons = current_layer._neurons_data->_num_neurons;

				float* values = current_layer._neurons_data->_values;

				assert(target_output.size() == num_neurons);

				float loss = 0.0f;

				//Calculate gradients
				for (size_t i = 0; i < num_neurons; i++)
				{
					loss += (target_output[i] - values[i]) * (target_output[i] - values[i]);
				}

				return sqrt(loss);
			}
			else
			{
				unsigned num_neurons = current_layer._neurons_CL_data->_num_neurons;

				Types::LayerValues values = BaseFunc::GetValues(current_layer);

				assert(target_output.size() == num_neurons);

				float loss = 0.0f;

				//Calculate gradients
				for (size_t i = 0; i < num_neurons; i++)
				{
					loss += (target_output[i] - values[i]) * (target_output[i] - values[i]);
				}

				return sqrt(loss);
			}
		}
	}
}