#include "NetBaseFunc.h"
extern "C" {
#include "OpenCLFunctions.h"
}

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

		void SetValues(LayerData& layer, Net_CLData* cl_data, Net_ArrayF values)
		{
			if (!layer._use_open_CL)
			{
				assert(layer._neurons_data != nullptr, "Neurons has not been init");
				assert(values._size == layer._neurons_data->_num_neurons);// , "Size of values, don't match the _size of _neurons, SetValus()");

				memcpy(layer._neurons_data->_values, values._data, sizeof(float) * layer._neurons_data->_num_neurons);
			}
			else
			{
				assert(layer._neurons_CL_data != nullptr, "Neurons has not been init");
				assert(values._size == layer._neurons_CL_data->_num_neurons);

				clEnqueueWriteBuffer(cl_data->_queue, layer._neurons_CL_data->_buffer_values, CL_FALSE, 0, sizeof(float) * layer._neurons_CL_data->_num_neurons, values._data, NULL, NULL, NULL);
				clFinish(cl_data->_queue);
			}
		}

		Net_ArrayF GetValues(LayerData& layer, Net_CLData* cl_data)
		{
			if (!layer._use_open_CL)
			{
				assert(layer._neurons_data != nullptr, "Neurons has not been init");

				Net_ArrayF values;

				Net_CreateArrayF(&values, layer._neurons_data->_num_neurons, 0);
				Net_MemCpyArrayF(&values, layer._neurons_data->_values, 0, layer._neurons_data->_num_neurons);

				return values;
			}
			else
			{
				assert(layer._neurons_CL_data != nullptr, "Neurons has not been init");

				Net_ArrayF values;

				Net_CreateArrayF(&values, layer._neurons_CL_data->_num_neurons, layer._neurons_CL_data->_num_neurons);

				clFinish(cl_data->_queue);//Make sure nothing is writing to buffer
				clEnqueueReadBuffer(cl_data->_queue, layer._neurons_CL_data->_buffer_values, CL_FALSE, 0, sizeof(float) * layer._neurons_CL_data->_num_neurons, values._data, NULL, NULL, NULL);
				clFinish(cl_data->_queue);

				return values;
			}
		}

		//Does not work with opencl
		void ActivateLayer(LayerData& layer)
		{
			float* values = layer._neurons_data->_values;

			switch (layer._function)
			{
			case Net_ActivationFunction::NET_ACTIVATION_FUNC_LEAKY_RELU:
				for (size_t i = 0; i < layer._neurons_data->_num_neurons; i++)
					values[i] = Net::Utility::Relu(values[i]);
				break;
			case Net_ActivationFunction::NET_ACTIVATION_FUNC_SIGMOID:
				for (size_t i = 0; i < layer._neurons_data->_num_neurons; i++)
					values[i] = Net::Utility::Sigmoid(values[i]);
				break;
			}
		}
	}
}