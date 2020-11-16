#include "NeuralNet.h"

#include "Layers/NetFCFunc.h"
#include "Layers/NetOutputFunc.h"
#include "Layers/NetConvFunc.h"
#include "NetBaseFunc.h"
#include "NetUtility.h"
#include "OpenCLFunctions.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

void Net_CreateNet(Net_NeuralNet* net, unsigned num_layers)
{
	net->_num_layers = num_layers;
	net->_num_init_layers = 0;
	net->_layers = malloc(sizeof(Net_LayerData) * num_layers);

	for (size_t i = 0; i < num_layers; i++)
	{
		net->_layers[i]._neurons_data = NULL;
		net->_layers[i]._neurons_CL_data = NULL;
		net->_layers[i]._weights_data = NULL;
		net->_layers[i]._weights_CL_data = NULL;
		net->_layers[i]._fC_layer_data = NULL;
		net->_layers[i]._conv_layer_data = NULL;
		net->_layers[i]._output_layer_CL_data = NULL;
	}

	Net_CLInitializeData(&net->_cl_data);
}

void Net_FreeLayersNet(Net_NeuralNet* net)
{
	for (size_t i = 0; i < net->_num_layers; i++)
	{
		cl_int error;

		if (net->_layers[i]._neurons_data != NULL)
		{
			if (net->_layers[i]._neurons_data->_values)
				free(net->_layers[i]._neurons_data->_values);
			if (net->_layers[i]._neurons_data->_gradient)
				free(net->_layers[i]._neurons_data->_gradient);

			free(net->_layers[i]._neurons_data);
		}
		if (net->_layers[i]._neurons_CL_data != NULL)
		{
			error = clReleaseMemObject(net->_layers[i]._neurons_CL_data->_buffer_values);
			assert(error == 0);
			error = clReleaseMemObject(net->_layers[i]._neurons_CL_data->_buffer_gradient);
			assert(error == 0);

			free(net->_layers[i]._neurons_CL_data);
		}
		if (net->_layers[i]._weights_data != NULL)
		{
			if (net->_layers[i]._weights_data->_weights)
				free(net->_layers[i]._weights_data->_weights);
			if (net->_layers[i]._weights_data->_delta_weights)
				free(net->_layers[i]._weights_data->_delta_weights);

			free(net->_layers[i]._weights_data);
		}
		if (net->_layers[i]._weights_CL_data != NULL)
		{
			error = clReleaseMemObject(net->_layers[i]._weights_CL_data->_buffer_weights);
			assert(error == 0);
			error = clReleaseMemObject(net->_layers[i]._weights_CL_data->_buffer_delta_weights);
			assert(error == 0);
			
			free(net->_layers[i]._weights_CL_data);
		}
		if (net->_layers[i]._fC_layer_data != NULL)
		{
			free(net->_layers[i]._fC_layer_data);
		}
		if (net->_layers[i]._conv_layer_data != NULL)
		{
			free(net->_layers[i]._conv_layer_data);
		}
		if (net->_layers[i]._output_layer_CL_data != NULL)
		{
			error = clReleaseMemObject(net->_layers[i]._output_layer_CL_data->_target_value);
			assert(error == 0);

			free(net->_layers[i]._output_layer_CL_data);
		}
	}

	Net_CLDeinitializeData(&net->_cl_data);
}

void Net_AddConvLayer(Net_NeuralNet* net, Net_ConvInitData* data, bool use_open_cl)
{
	assert(net->_num_init_layers < net->_num_layers);//"Can't add more layers"

	Net_LayerData* layer = &net->_layers[net->_num_init_layers];

	const unsigned num_neurons = data->_length * data->_length * data->_depth;//TODO: Do we need a bias in conv layer?
	const unsigned num_neurons_next = ((data->_length - data->_filter_length) / data->_stride + 1) * ((data->_length - data->_filter_length) / data->_stride + 1) * data->_num_filters;
	const unsigned num_weights = data->_filter_length * data->_filter_length * data->_depth * data->_num_filters;

	layer->_conv_layer_data = malloc(sizeof(Net_CompConvLayer));

	layer->_type = NET_TYPE_CONVOLUTIONAL;
	layer->_function = data->_function;
	layer->_use_open_CL = use_open_cl;

	layer->_conv_layer_data->_depth = data->_depth;
	layer->_conv_layer_data->_filter_length = data->_filter_length;
	layer->_conv_layer_data->_length = data->_length;
	layer->_conv_layer_data->_num_filters = data->_num_filters;
	layer->_conv_layer_data->_stride = data->_stride;

	float* weights = malloc(sizeof(float) *num_weights);

	for (size_t i = 0; i < num_weights; i++)
	{
		//Todo: fix so weight init is correct for every activation function. It's currently using the best known for relu
		weights[i] = Net_UtilRandomizeOneToZero() * (2.0f / (data->_filter_length * data->_filter_length * data->_depth));
	}

	if (use_open_cl)
	{
		layer->_neurons_CL_data = malloc(sizeof(Net_CompNeuronsCL));
		layer->_weights_CL_data = malloc(sizeof(Net_CompWeightsCL));

		layer->_neurons_CL_data->_num_neurons = num_neurons;
		layer->_neurons_CL_data->_num_neurons_next = num_neurons_next;

		layer->_weights_CL_data->_num_weights = num_weights;

		float* values = calloc(num_neurons, sizeof(float));//TODO: Maybe there is a better way to initialize values to zero in the buffers?
		float* delta_weights = calloc(num_weights, sizeof(float));
		
		cl_int error = 0;
		layer->_neurons_CL_data->_buffer_values = clCreateBuffer(net->_cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_neurons, values, &error);
		assert(error == 0);
		layer->_neurons_CL_data->_buffer_gradient = clCreateBuffer(net->_cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_neurons, values, &error);
		assert(error == 0);
		layer->_weights_CL_data->_buffer_weights = clCreateBuffer(net->_cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_weights, weights, &error);
		assert(error == 0);
		layer->_weights_CL_data->_buffer_delta_weights = clCreateBuffer(net->_cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_weights, delta_weights, &error);
		assert(error == 0);

		free(weights);
		free(values);
		free(delta_weights);
	}
	else
	{
		layer->_neurons_data = malloc(sizeof(Net_CompNeurons));
		layer->_weights_data = malloc(sizeof(Net_CompWeights));

		layer->_neurons_data->_num_neurons = num_neurons;
		layer->_neurons_data->_num_neurons_next = num_neurons_next;

		layer->_weights_data->_num_weights = num_weights;

		layer->_neurons_data->_values = calloc(num_neurons, sizeof(float));
		layer->_neurons_data->_gradient = calloc(num_neurons, sizeof(float));
		layer->_weights_data->_delta_weights = calloc(num_weights, sizeof(float));
		layer->_weights_data->_weights = weights;//Takes ownership of pointer
	}

	net->_num_init_layers++;
}

void Net_AddFCLayer(Net_NeuralNet* net, Net_FCInitData* data, bool use_open_cl)
{
	assert(net->_num_init_layers < net->_num_layers);//"Can't add more layers"

	Net_LayerData* layer = &net->_layers[net->_num_init_layers];

	const unsigned num_neurons_with_bias = data->_num_neurons + 1;
	const unsigned num_weights = num_neurons_with_bias * data->_num_neurons_next;

	layer->_fC_layer_data = malloc(sizeof(Net_CompFCLayer));

	layer->_type = NET_TYPE_FULLY_CONNECTED;
	layer->_function = data->_function;
	layer->_use_open_CL = use_open_cl;

	layer->_fC_layer_data->_num_neurons_with_bias = num_neurons_with_bias;

	float* weights = malloc(sizeof(float) *num_weights);

	for (size_t i = 0; i < num_weights; i++)
	{
		//Todo: fix so weight init is correct for every activation function. It's currently using the best known for relu
		weights[i] = Net_UtilRandomizeOneToZero() * (2.0f / num_neurons_with_bias);
	}

	if (use_open_cl)
	{
		layer->_neurons_CL_data = malloc(sizeof(Net_CompNeuronsCL));
		layer->_weights_CL_data = malloc(sizeof(Net_CompWeightsCL));

		layer->_neurons_CL_data->_num_neurons = data->_num_neurons;
		layer->_neurons_CL_data->_num_neurons_next = data->_num_neurons_next;

		layer->_weights_CL_data->_num_weights = num_weights;
		
		float* values = calloc(num_neurons_with_bias, sizeof(float));
		float* delta_weights = calloc(num_weights, sizeof(float));

		values[num_neurons_with_bias - 1] = 1.0f;//Set bias

		cl_int error = 0;
		layer->_neurons_CL_data->_buffer_values = clCreateBuffer(net->_cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_neurons_with_bias, values, &error);
		assert(error == 0);
		layer->_neurons_CL_data->_buffer_gradient = clCreateBuffer(net->_cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_neurons_with_bias, values, &error);
		assert(error == 0);
		layer->_weights_CL_data->_buffer_weights = clCreateBuffer(net->_cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_weights, weights, &error);
		assert(error == 0);
		layer->_weights_CL_data->_buffer_delta_weights = clCreateBuffer(net->_cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_weights, delta_weights, &error);
		assert(error == 0);

		free(weights);
		free(values);
		free(delta_weights);
	}
	else
	{
		layer->_neurons_data = malloc(sizeof(Net_CompNeurons));
		layer->_weights_data = malloc(sizeof(Net_CompWeights));

		layer->_neurons_data->_num_neurons = data->_num_neurons;
		layer->_neurons_data->_num_neurons_next = data->_num_neurons_next;

		layer->_weights_data->_num_weights = num_weights;

		layer->_neurons_data->_values = calloc(num_neurons_with_bias, sizeof(float));
		layer->_neurons_data->_gradient = calloc(num_neurons_with_bias, sizeof(float));//TODO: Do we need gradient for bias?
		layer->_weights_data->_delta_weights = calloc(num_weights, sizeof(float));
		layer->_weights_data->_weights = weights;//Takes ownership

		layer->_neurons_data->_values[num_neurons_with_bias - 1] = 1.0f;//Set bias
	}

	net->_num_init_layers++;
}

void Net_AddOutputLayer(Net_NeuralNet* net, Net_OutputInitData* data, bool use_open_cl)
{
	assert(net->_num_init_layers < net->_num_layers);//"Can't add more layers"

	Net_LayerData* layer = &net->_layers[net->_num_init_layers];

	layer->_type = NET_TYPE_OUTPUT;
	layer->_function = data->_function;
	layer->_use_open_CL = use_open_cl;

	if (use_open_cl)
	{
		layer->_neurons_CL_data = malloc(sizeof(Net_CompNeuronsCL));
		layer->_output_layer_CL_data = malloc(sizeof(Net_CompOutLayerCL));

		layer->_neurons_CL_data->_num_neurons = data->_num_neurons;

		float* values = calloc(data->_num_neurons, sizeof(float));//TODO: Maybe there is a beter way to initilize valus to zero in the buffers?

		cl_int error = 0;
		layer->_neurons_CL_data->_buffer_values = clCreateBuffer(net->_cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * data->_num_neurons, values, &error);
		assert(error == 0);
		layer->_neurons_CL_data->_buffer_gradient = clCreateBuffer(net->_cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * data->_num_neurons, values, &error);
		assert(error == 0);
		layer->_output_layer_CL_data->_target_value = clCreateBuffer(net->_cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * data->_num_neurons, values, &error);
		assert(error == 0);

		free(values);
	}
	else
	{
		layer->_neurons_data = malloc(sizeof(Net_CompNeurons));

		layer->_neurons_data->_num_neurons = data->_num_neurons;

		layer->_neurons_data->_values = calloc(data->_num_neurons, sizeof(float));
		layer->_neurons_data->_gradient = calloc(data->_num_neurons, sizeof(float));
	}

	net->_num_init_layers++;
}

void Net_BackpropUseCurrentState(Net_NeuralNet* net, const Net_ArrayF* target_output, float eta, float momentum)
{
	for (int i = net->_num_layers - 1; i >= 0; i--)
	{
		Net_LayerData* layer = &net->_layers[i];
		Net_LayerData* next_layer = &net->_layers[i + 1];

		switch (layer->_type)
		{
		case NET_TYPE_FULLY_CONNECTED:
			Net_FCBackprop(layer, next_layer, &net->_cl_data, eta, momentum);
			break;
		case NET_TYPE_CONVOLUTIONAL:
			Net_ConvBackprop(layer, next_layer, &net->_cl_data, eta, momentum);
			break;
		case NET_TYPE_OUTPUT:
			Net_OutBackprop(layer, target_output, &net->_cl_data);
			break;
		}
	}
}

float Net_CalcLoss(Net_NeuralNet* net, const Net_ArrayF* target_output)
{
	return Net_OutGetLoss(&net->_layers[net->_num_layers - 1], target_output, &net->_cl_data);
}

void Net_FeedForward(Net_NeuralNet* net, Net_ArrayF input)
{
	//Set values of the input layer
	Net_BaseSetValues(&net->_layers[0], &net->_cl_data, input);

	for (unsigned i = 0; i < net->_num_layers - 1; i++)
	{
		Net_LayerData* layer = &net->_layers[i];
		Net_LayerData* next_layer = &net->_layers[i + 1];

		switch (layer->_type)
		{
		case NET_TYPE_FULLY_CONNECTED:
			Net_FCFeedForward(layer, next_layer, &net->_cl_data);
			break;
		case NET_TYPE_CONVOLUTIONAL:
			Net_ConvFeedForward(layer, next_layer, &net->_cl_data);
			break;
		}
	}
}

Net_ArrayF Net_GetOutputValues(Net_NeuralNet* net)
{
	return Net_BaseGetValues(&net->_layers[net->_num_layers - 1], &net->_cl_data);
}

//TODO: Update
//void ValidateNet(NeuralNet& net)
//{
//
//	for (size_t i = 0; i < net._num_layers - 1; i++)
//	{
//		BaseLayer& layer = net._layers[i];
//		BaseLayer& next_layer = net._layers[i + 1];
//
//		assert(layer._num_neurons_next == next_layer._num_neurons_no_bias);
//	}
//	for (size_t i = 0; i < net._num_layers; i++)
//	{
//
//		//TODO check that stride make sence/ if zero
//		BaseLayer& layer = net._layers[i];
//
//		if (layer._type == BaseLayer::NET_TYPE_CONVOLUTIONAL)
//		{
//			assert(layer._filter_length % 2 == 1);
//		}
//
//	}
//
//	for (size_t i = 0; i < net._num_layers; i++)
//	{
//		BaseLayer& layer = net._layers[i];
//
//		for (size_t j = 0; j < layer._num_weights; j++)
//		{
//			assert(layer._weights[j] < 10000.0f && layer._weights[j] > -10000.0f);
//		}
//		for (size_t j = 0; j < layer._num_weights; j++)
//		{
//			assert(layer._delta_weights[j] < 10000.0f && layer._delta_weights[j] > -10000.0f);
//		}
//		for (size_t j = 0; j < layer._num_neurons_no_bias; j++)
//		{
//			layer._values[j] = 0.0f;
//		}
//		for (size_t j = 0; j < layer._num_neurons; j++)
//		{
//			layer._gradient[j] = 0.0f;
//		}
//
//		if (layer._type == NetLayer::NetType::NET_TYPE_FULLY_CONNECTED) {
//			assert(layer._values[layer._num_neurons - 1] == 1.0f);
//		}
//	}
//}

bool Net_SaveNet(const Net_NeuralNet* net, const char* path)
{
	unsigned buffer_size = (net->_num_layers + 1) * sizeof(unsigned);// store num layers and num neurons

	for (size_t i = 0; i < net->_num_layers; i++)
	{
		if (!net->_layers[i]._weights_data && !net->_layers[i]._weights_CL_data)
			continue;

		if (net->_layers[i]._use_open_CL)
		{
			buffer_size += net->_layers[i]._weights_CL_data->_num_weights * 2 * sizeof(float);//+ 1 to sore size
		}
		else
		{
			buffer_size += net->_layers[i]._weights_data->_num_weights * 2 * sizeof(float);//+ 1 to sore size
		}
	}

	unsigned offset = 0;
	char* data = malloc(buffer_size);	

	memcpy(&data[offset], &net->_num_layers, sizeof(unsigned));
	offset += sizeof(unsigned);

	for (size_t i = 0; i < net->_num_layers; i++)
	{
		const Net_LayerData* layer = &net->_layers[i];

		if (!layer->_weights_data && !layer->_weights_CL_data)
			continue;

		if (layer->_use_open_CL)
		{
			unsigned num_weights = layer->_weights_CL_data->_num_weights;

			memcpy(&data[offset], &layer->_weights_CL_data->_num_weights, sizeof(unsigned));
			offset += sizeof(unsigned);
			
			cl_int error = 0;
			
			error = clFinish(net->_cl_data._queue);
			assert(error == 0);
			
			error = clEnqueueReadBuffer(net->_cl_data._queue, layer->_weights_CL_data->_buffer_weights, CL_FALSE, 0, sizeof(float) * num_weights, &data[offset], NULL, NULL, NULL);
			offset += layer->_weights_CL_data->_num_weights * sizeof(float);
			assert(error == 0);
			
			error = clEnqueueReadBuffer(net->_cl_data._queue, layer->_weights_CL_data->_buffer_delta_weights, CL_FALSE, 0, sizeof(float) * num_weights, &data[offset], NULL, NULL, NULL);
			offset += layer->_weights_CL_data->_num_weights * sizeof(float);
			assert(error == 0);
			
			error = clFinish(net->_cl_data._queue);
			assert(error == 0);
		}
		else
		{
			memcpy(&data[offset], &layer->_weights_data->_num_weights, sizeof(unsigned));
			offset += sizeof(unsigned);

			memcpy(&data[offset], layer->_weights_data->_weights, layer->_weights_data->_num_weights * sizeof(float));
			offset += layer->_weights_data->_num_weights * sizeof(float);
			memcpy(&data[offset], layer->_weights_data->_delta_weights, layer->_weights_data->_num_weights * sizeof(float));
			offset += layer->_weights_data->_num_weights * sizeof(float);
		}
	}

	FILE* file;

	fopen_s(&file, path, "wb");

	assert(file != NULL);

	fwrite(data, sizeof(char), buffer_size, file);

	fclose(file);
	free(data);
}

bool Net_LoadNet(Net_NeuralNet* net, const char* path)
{
	//Load file
	FILE* file;

	fopen_s(&file, path, "rb");

	if (file == NULL)
		return false;

	unsigned offset = 0;

	fseek(file, 0, SEEK_END);
	long buffer_size = ftell(file);
	rewind(file);

	char* data = (char*)malloc(buffer_size);

	fread_s(data, buffer_size, buffer_size, 1, file);
	fclose(file);
	
	unsigned num_layers;
	memcpy(&num_layers, &data[offset], sizeof(unsigned));
	offset += sizeof(unsigned);

	assert(net->_num_layers == num_layers);

	for (size_t i = 0; i < net->_num_layers; i++)
	{
		const Net_LayerData* layer = &net->_layers[i];
		
		if (!layer->_weights_data && !layer->_weights_CL_data)
			continue;

		unsigned num_weights;
		memcpy(&num_weights, &data[offset], sizeof(unsigned));
		offset += sizeof(unsigned);

		if (layer->_use_open_CL)
		{
			assert(num_weights == layer->_weights_CL_data->_num_weights);
			const int size = sizeof(float) * layer->_weights_CL_data->_num_weights;
			
			cl_int error = 0;
		
			error = clFinish(net->_cl_data._queue);
			assert(error == 0);
		
			error = clEnqueueWriteBuffer(net->_cl_data._queue, layer->_weights_CL_data->_buffer_weights, CL_FALSE, 0, size, &data[offset], NULL, NULL, NULL);
			offset += num_weights * sizeof(float);
			assert(error == 0);
			
			error = clEnqueueWriteBuffer(net->_cl_data._queue, layer->_weights_CL_data->_buffer_delta_weights, CL_FALSE, 0, size, &data[offset], NULL, NULL, NULL);
			offset += num_weights * sizeof(float);
			assert(error == 0);
		
			error = clFinish(net->_cl_data._queue);
			assert(error == 0);
		}
		else
		{
			assert(num_weights == layer->_weights_data->_num_weights);

			const int size = sizeof(float) * num_weights;
		
			memcpy(layer->_weights_data->_weights, &data[offset], size);
			offset += num_weights * sizeof(float);
			memcpy(layer->_weights_data->_delta_weights, &data[offset], size);
			offset += num_weights * sizeof(float);
		}
	}

	free(data);

	return true;

	//std::ifstream stream(path, std::ifstream::binary);
	//std::string data(std::istreambuf_iterator<char>(stream), (std::istreambuf_iterator<char>()));
	//
	//if (!data[0]) 
	//{
	//	std::cout << "Can't load data, LoadNet()" << std::endl;
	//	return false;
	//}
	//
	//const Serialization::NetData* net_data = Serialization::GetNetData(data.data());
	//
	//for (unsigned i = 0; i < net_data->_layer()->size(); i++)
	//{
	//	Net_LayerData* layer = &net->_layers[i];
	//
	//	//Output layer have no weights
	//	if (!layer->_weights_data && !layer->_weights_CL_data)
	//		continue;
	//
	//	const Serialization::Layer* data_layer = net_data->_layer()->Get(i);
	//
	//	if (layer->_use_open_CL)
	//	{
	//		const int _size = sizeof(float) * layer->_weights_CL_data->_num_weights;
	//		
	//		cl_int error = 0;
	//
	//		error = clFinish(net->_cl_data._queue);
	//		assert(error == 0);
	//
	//		error = clEnqueueWriteBuffer(net->_cl_data._queue, layer->_weights_CL_data->_buffer_weights, CL_FALSE, 0, _size, data_layer->_weight()->Data(), NULL, NULL, NULL);
	//		assert(error == 0);
	//		error = clEnqueueWriteBuffer(net->_cl_data._queue, layer->_weights_CL_data->_buffer_delta_weights, CL_FALSE, 0, _size, data_layer->_delta_weight()->Data(), NULL, NULL, NULL);
	//		assert(error == 0);
	//
	//		error = clFinish(net->_cl_data._queue);
	//		assert(error == 0);
	//	}
	//	else
	//	{
	//		const int _size = sizeof(float) * layer->_weights_data->_num_weights;
	//
	//		memcpy(layer->_weights_data->_weights, data_layer->_weight()->Data(), _size);
	//		memcpy(layer->_weights_data->_delta_weights, data_layer->_delta_weight()->Data(), _size);
	//	}
	//}
	//
	//return true;
}

void Net_SaveTrainingData(Net_ArrayTrainingData* training_data, const char* path)
{
	assert(training_data != NULL);
	assert(training_data->_data[0]._input._data != NULL);
	assert(training_data->_data[0]._output._data != NULL);

	const unsigned num_training_data = training_data->_size;
	const unsigned input_size = training_data->_data[0]._input._size * sizeof(float);
	const unsigned output_size = training_data->_data[0]._output._size * sizeof(float);
	const unsigned buffer_size = (num_training_data * input_size + num_training_data * output_size) * sizeof(float) + 3 * sizeof(unsigned);

	char* data = malloc(buffer_size);

	unsigned offset = 0;

	memcpy(&data[offset], &num_training_data, sizeof(unsigned));
	offset += sizeof(unsigned);
	memcpy(&data[offset], &training_data->_data[0]._input._size, sizeof(unsigned));
	offset += sizeof(unsigned);
	memcpy(&data[offset], &training_data->_data[0]._output._size, sizeof(unsigned));
	offset += sizeof(unsigned);

	FILE* file;

	fopen_s(&file, path, "wb");

	assert(file != NULL);

	for (size_t i = 0; i < num_training_data; i++)
	{
		memcpy(&data[offset], training_data->_data[i]._input._data, input_size);
		offset += input_size;
		memcpy(&data[offset], training_data->_data[i]._output._data, output_size);
		offset += output_size;
	}

	fwrite(data, sizeof(char), buffer_size, file);

	fclose(file);
	free(data);
}

Net_ArrayTrainingData Net_LoadTrainingData(const char* path)//TODO: Maybe pass training _data as ref and return bool true if successful?
{
	//Load file
	FILE* file;

	fopen_s(&file, path, "rb");

	assert(file != NULL);

	unsigned num_training_data;
	unsigned input_size;
	unsigned output_size;
	unsigned buffer_size;

	fread_s(&num_training_data, sizeof(unsigned), sizeof(unsigned), 1, file);
	fread_s(&input_size, sizeof(unsigned), sizeof(unsigned), 1, file);
	fread_s(&output_size, sizeof(unsigned), sizeof(unsigned), 1, file);
	buffer_size = (num_training_data * input_size + num_training_data * output_size) * sizeof(float) + 3 * sizeof(unsigned);

	char* data = (char*)malloc(buffer_size);

	fread_s(data, buffer_size, buffer_size, 1, file);
	fclose(file);

	Net_ArrayTrainingData training_data;
	Net_CreateArrayTD(&training_data, num_training_data, num_training_data);

	unsigned offset = 12;

	for (size_t i = 0; i < num_training_data; i++)
	{
		Net_CreateArrayF(&training_data._data[i]._input, input_size, input_size);
		Net_MemCpyArrayF(&training_data._data[i]._input, &data[offset], 0, input_size);
		offset += input_size * sizeof(float);

		Net_CreateArrayF(&training_data._data[i]._output, output_size, output_size);
		Net_MemCpyArrayF(&training_data._data[i]._output, &data[offset], 0, output_size);
		offset += output_size * sizeof(float);
	}

	free(data);

	return training_data;
}

void Net_TrainNet(Net_NeuralNet* net, const Net_ArrayTrainingData* training_data, float eta, float momentum, int display_info_rate)
{
	int display_info_in = display_info_rate - 1;

	float loss = 0.0f;

	for (size_t i = 0; i < training_data->_size; i++)
	{
		Net_FeedForward(net, training_data->_data[i]._input);
		Net_BackpropUseCurrentState(net, &training_data->_data[i]._output, eta, momentum);

		loss += Net_CalcLoss(net, &training_data->_data[i]._output);

		if (display_info_in <= 0 && display_info_rate != 0)
		{
			printf("Num backprop: %zu", i + 1u);
			printf("  Loss: %f \n", loss / (float)display_info_rate);
			loss = 0.0f;
			display_info_in = display_info_rate - 1;
		}
		else
		{
			display_info_in--;
		}
	}
}