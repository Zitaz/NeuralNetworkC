#include "NetBaseFunc.h"
#include "OpenCLFunctions.h"

#include <assert.h>

namespace Net
{
	namespace BaseFunc
	{
		void SetValuesToZero(LayerData& layer)
		{
			assert(layer._neurons_data != nullptr, "Neurons has not been init");

			memset(layer._neurons_data->_values, 0x00, layer._neurons_data->_num_neurons);
		}

		void SetValues(LayerData& layer, Types::LayerValues values)
		{
			if (!layer._use_open_CL)
			{
				assert(layer._neurons_data != nullptr, "Neurons has not been init");
				assert(values.size() == layer._neurons_data->_num_neurons);// , "Size of values, don't match the size of _neurons, SetValus()");

				memcpy(layer._neurons_data->_values, values.data(), sizeof(float) * layer._neurons_data->_num_neurons);
			}
			else
			{
				assert(layer._neurons_CL_data != nullptr, "Neurons has not been init");
				assert(values.size() == layer._neurons_CL_data->_num_neurons);

				clEnqueueWriteBuffer(OpenCL::Data::instace.queue, layer._neurons_CL_data->_buffer_values, CL_FALSE, 0, sizeof(float) * layer._neurons_CL_data->_num_neurons, &values[0], NULL, NULL, NULL);
				clFinish(OpenCL::Data::instace.queue);
			}
		}

		Types::LayerValues GetValues(LayerData& layer)
		{
			if (!layer._use_open_CL)
			{
				assert(layer._neurons_data != nullptr, "Neurons has not been init");

				Types::LayerValues values(layer._neurons_data->_num_neurons);

				memcpy(values.data(), layer._neurons_data->_values, sizeof(float) * layer._neurons_data->_num_neurons);

				return values;
			}
			else
			{
				assert(layer._neurons_CL_data != nullptr, "Neurons has not been init");

				Types::LayerValues values(layer._neurons_CL_data->_num_neurons);

				clFinish(OpenCL::Data::instace.queue);//Make sure nothing is writing to buffer
				clEnqueueReadBuffer(OpenCL::Data::instace.queue, layer._neurons_CL_data->_buffer_values, CL_FALSE, 0, sizeof(float) * layer._neurons_CL_data->_num_neurons, values.data(), NULL, NULL, NULL);
				clFinish(OpenCL::Data::instace.queue);

				return values;
			}
		}

		//Does not work with opencl
		void ActivateLayer(LayerData& layer)
		{
			float* values = layer._neurons_data->_values;

			switch (layer._function)
			{
			case Types::ActivationFunction::LEAKY_RELU:
				for (size_t i = 0; i < layer._neurons_data->_num_neurons; i++)
					values[i] = Net::Utility::Relu(values[i]);
				break;
			case Types::ActivationFunction::SIGMOID:
				for (size_t i = 0; i < layer._neurons_data->_num_neurons; i++)
					values[i] = Net::Utility::Sigmoid(values[i]);
				break;
			}
		}
	}
}