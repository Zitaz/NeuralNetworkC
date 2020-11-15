#include "NeuralNet.h"
#include "Layers/NetConvFunc.h"
#include "Layers/NetOutputFunc.h"
#include "Layers/NetFCFunc.h"
#include "NetBaseFunc.h"
#include "NetUtility.h"

extern "C" {
#include "OpenCLFunctions.h"
}

#include "flatbuffers.h"
#include "Generated/NetData_generated.h"
#include "Generated/TrainingData_generated.h"
#include "util.h"

#include <assert.h>
#include <math.h>
#include <iostream>

namespace Net
{

	NeuralNet::~NeuralNet()
	{
		if (_layers)
			delete[] _layers;
	}

	namespace NetFunc
	{
		void CreateNeuralNet(NeuralNet* net, unsigned num_layers)
		{
			net->_num_layers = num_layers;
			net->_num_init_layers = 0;
			net->_layers = new LayerData[num_layers];
			Net_CLInitializeData(&net->_cl_data);
		}

		void AddConvLayer(NeuralNet& net, InitData::NetConvInitData& _data, bool use_open_cl)//TODO: Can probably split into more function for reuse
		{
			assert(net._num_init_layers < net._num_layers);//"Can't add more layers"

			LayerData& layer = net._layers[net._num_init_layers];

			const unsigned num_neurons = _data._length * _data._length * _data._depth;//TODO: Do we need a bias in conv layer?
			const unsigned num_neurons_next = ((_data._length - _data._filter_length) / _data._stride + 1) * ((_data._length - _data._filter_length) / _data._stride + 1) * _data._num_filters;
			const unsigned num_weights = _data._filter_length * _data._filter_length * _data._depth * _data._num_filters;

			layer._conv_layer_data = new ConvLayerData;

			layer._type = Net_NetType::NET_TYPE_CONVOLUTIONAL;
			layer._function = _data._function;
			layer._use_open_CL = use_open_cl;

			layer._conv_layer_data->_depth = _data._depth;
			layer._conv_layer_data->_filter_length = _data._filter_length;
			layer._conv_layer_data->_length = _data._length;
			layer._conv_layer_data->_num_filters = _data._num_filters;
			layer._conv_layer_data->_stride = _data._stride;

			float* weights = new float[num_weights];

			for (size_t i = 0; i < num_weights; i++)
			{
				//Todo: fix so weight init is correct for every activation function. It's currently using the best known for relu
				weights[i] = Utility::RandomizeOneToZero() * (2.0f / (_data._filter_length * _data._filter_length * _data._depth));
			}

			if (use_open_cl)
			{
				layer._neurons_CL_data = new NeuronsCLData;
				layer._weights_CL_data = new WeightsCLData;

				layer._neurons_CL_data->_num_neurons = num_neurons;
				layer._neurons_CL_data->_num_neurons_next = num_neurons_next;

				layer._weights_CL_data->_num_weights = num_weights;

				float* values = new float[num_neurons] { 0 };//TODO: Maybe there is a better way to initialize values to zero in the buffers?
				float* delta_weights = new float[num_weights] { 0 };
				
				cl_int error = 0;
				layer._neurons_CL_data->_buffer_values = clCreateBuffer(net._cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_neurons, values, &error);
				assert(error == 0);
				layer._neurons_CL_data->_buffer_gradient = clCreateBuffer(net._cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_neurons, values, &error);
				assert(error == 0);
				layer._weights_CL_data->_buffer_weights = clCreateBuffer(net._cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_weights, weights, &error);
				assert(error == 0);
				layer._weights_CL_data->_buffer_delta_weights = clCreateBuffer(net._cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_weights, delta_weights, &error);
				assert(error == 0);

				delete[] weights;
				delete[] values;
				delete[] delta_weights;
			}
			else
			{
				layer._neurons_data = new NeuronsData;
				layer._weights_data = new WeightsData;

				layer._neurons_data->_num_neurons = num_neurons;
				layer._neurons_data->_num_neurons_next = num_neurons_next;

				layer._weights_data->_num_weights = num_weights;

				layer._neurons_data->_values = new float[num_neurons] { 0 };
				layer._neurons_data->_gradient = new float[num_neurons] { 0 };
				layer._weights_data->_delta_weights = new float[num_weights] { 0 };
				layer._weights_data->_weights = weights;//Takes ownership of pointer
			}

			net._num_init_layers++;
		}

		void AddFCLayer(NeuralNet& net, InitData::NetFCInitData& _data, bool use_open_cl)
		{
			assert(net._num_init_layers < net._num_layers);//"Can't add more layers"

			LayerData& layer = net._layers[net._num_init_layers];

			const unsigned num_neurons_with_bias = _data._num_neurons + 1;
			const unsigned num_weights = num_neurons_with_bias * _data._num_neurons_next;

			layer._fC_layer_data = new FCLayerData;

			layer._type = Net_NetType::NET_TYPE_FULLY_CONNECTED;
			layer._function = _data._function;
			layer._use_open_CL = use_open_cl;

			layer._fC_layer_data->_num_neurons_with_bias = num_neurons_with_bias;

			float* weights = new float[num_weights];

			for (size_t i = 0; i < num_weights; i++)
			{
				//Todo: fix so weight init is correct for every activation function. It's currently using the best known for relu
				weights[i] = Utility::RandomizeOneToZero() * (2.0f / num_neurons_with_bias);
			}

			if (use_open_cl)
			{
				layer._neurons_CL_data = new NeuronsCLData;
				layer._weights_CL_data = new WeightsCLData;

				layer._neurons_CL_data->_num_neurons = _data._num_neurons;
				layer._neurons_CL_data->_num_neurons_next = _data._num_neurons_next;

				layer._weights_CL_data->_num_weights = num_weights;

				float* values = new float[num_neurons_with_bias] { 0 };
				float* delta_weights = new float[num_weights] { 0 };

				values[num_neurons_with_bias - 1] = 1.0f;//Set bias

				cl_int error = 0;
				layer._neurons_CL_data->_buffer_values = clCreateBuffer(net._cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_neurons_with_bias, values, &error);
				assert(error == 0);
				layer._neurons_CL_data->_buffer_gradient = clCreateBuffer(net._cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_neurons_with_bias, values, &error);
				assert(error == 0);
				layer._weights_CL_data->_buffer_weights = clCreateBuffer(net._cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_weights, weights, &error);
				assert(error == 0);
				layer._weights_CL_data->_buffer_delta_weights = clCreateBuffer(net._cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_weights, delta_weights, &error);
				assert(error == 0);

				delete[] weights;
				delete[] values;
				delete[] delta_weights;
			}
			else
			{
				layer._neurons_data = new NeuronsData;
				layer._weights_data = new WeightsData;

				layer._neurons_data->_num_neurons = _data._num_neurons;
				layer._neurons_data->_num_neurons_next = _data._num_neurons_next;

				layer._weights_data->_num_weights = num_weights;

				layer._neurons_data->_values = new float[num_neurons_with_bias] { 0 };
				layer._neurons_data->_gradient = new float[num_neurons_with_bias] { 0 };//TODO: Do we need gradient for bias?
				layer._weights_data->_delta_weights = new float[num_weights] { 0 };
				layer._weights_data->_weights = weights;//Takes ownership

				layer._neurons_data->_values[num_neurons_with_bias - 1] = 1.0f;//Set bias
			}

			net._num_init_layers++;
		}

		void AddOutputLayer(NeuralNet& net, InitData::NetOutputInitData& _data, bool use_open_cl)
		{
			assert(net._num_init_layers < net._num_layers);//"Can't add more layers"

			LayerData& layer = net._layers[net._num_init_layers];

			layer._type = Net_NetType::NET_TYPE_OUTPUT;
			layer._function = _data._function;
			layer._use_open_CL = use_open_cl;

			if (use_open_cl)
			{
				layer._neurons_CL_data = new NeuronsCLData;
				layer._output_layer_CL_data = new OutputLayerCLData;

				layer._neurons_CL_data->_num_neurons = _data._num_neurons;

				float* values = new float[_data._num_neurons]{ 0 };//TODO: Maybe there is a beter way to initilize valus to zero in the buffers?

				cl_int error = 0;
				layer._neurons_CL_data->_buffer_values = clCreateBuffer(net._cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * _data._num_neurons, values, &error);
				assert(error == 0);
				layer._neurons_CL_data->_buffer_gradient = clCreateBuffer(net._cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * _data._num_neurons, values, &error);
				assert(error == 0);
				layer._output_layer_CL_data->_target_value = clCreateBuffer(net._cl_data._context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * _data._num_neurons, values, &error);
				assert(error == 0);

				delete[] values;
			}
			else
			{
				layer._neurons_data = new NeuronsData;

				layer._neurons_data->_num_neurons = _data._num_neurons;

				layer._neurons_data->_values = new float[_data._num_neurons]{ 0 };
				layer._neurons_data->_gradient = new float[_data._num_neurons]{ 0 };
			}

			net._num_init_layers++;
		}

		void BackpropUseCurrentState(NeuralNet& net, const Net_ArrayF& target_output, float eta, float momentum)
		{
			for (int i = net._num_layers - 1; i >= 0; i--)
			{
				LayerData& layer = net._layers[i];
				LayerData& next_layer = net._layers[i + 1];

				switch (layer._type)
				{
				case Net_NetType::NET_TYPE_FULLY_CONNECTED:
					FCFunc::Backprop(layer, next_layer, &net._cl_data, eta, momentum);
					break;
				case Net_NetType::NET_TYPE_CONVOLUTIONAL:
					ConvFunc::Backprop(layer, next_layer, &net._cl_data, eta, momentum);
					break;
				case Net_NetType::NET_TYPE_OUTPUT:
					OutputFunc::Backprop(layer, target_output, &net._cl_data);
					break;
				}
			}
		}

		float CalcLoss(NeuralNet& net, const Net_ArrayF& target_output)
		{
			return OutputFunc::GetLoss(net._layers[net._num_layers - 1], target_output, &net._cl_data);
		}

		void FeedForward(NeuralNet& net, Net_ArrayF input)
		{
			//Set values of the input layer
			BaseFunc::SetValues(net._layers[0], &net._cl_data, input);

			for (unsigned i = 0; i < net._num_layers - 1; i++)
			{
				LayerData& layer = net._layers[i];
				LayerData& next_layer = net._layers[i + 1];

				switch (layer._type)
				{
				case Net_NetType::NET_TYPE_FULLY_CONNECTED:
					FCFunc::FeedForward(layer, next_layer, &net._cl_data);
					break;
				case Net_NetType::NET_TYPE_CONVOLUTIONAL:
					ConvFunc::FeedForward(layer, next_layer, &net._cl_data);
					break;
				}
			}
		}

		Net_ArrayF GetOutputValues(NeuralNet& net)
		{
			return BaseFunc::GetValues(net._layers[net._num_layers - 1], &net._cl_data);
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
		
		bool SaveNet(const NeuralNet& net, const char* path)
		{
			flatbuffers::FlatBufferBuilder builder(1024);
		
			std::vector<flatbuffers::Offset<Serialization::Layer>> layers;
		
			for (size_t i = 0; i < net._num_layers; i++)
			{
				const LayerData& layer = net._layers[i];
		
				//Output layer have no weights
				if (!layer._weights_data && !layer._weights_CL_data)
					continue;
		

				unsigned num_weights;

				flatbuffers::Offset<flatbuffers::Vector<float>> weights;
				flatbuffers::Offset<flatbuffers::Vector<float>> delta_weights;
		
				if (layer._use_open_CL)
				{
					num_weights = layer._weights_CL_data->_num_weights;

					cl_int error = 0;
		
					float* layer_weights = new float[num_weights];
					float* layer_delta_weights = new float[num_weights];
		
					error = clFinish(net._cl_data._queue);
					assert(error == 0);

					error = clEnqueueReadBuffer(net._cl_data._queue, layer._weights_CL_data->_buffer_weights, CL_FALSE, 0, sizeof(float) * num_weights, layer_weights, NULL, NULL, NULL);
					assert(error == 0);
					error = clEnqueueReadBuffer(net._cl_data._queue, layer._weights_CL_data->_buffer_delta_weights, CL_FALSE, 0, sizeof(float) * num_weights, layer_delta_weights, NULL, NULL, NULL);
					assert(error == 0);

					error = clFinish(net._cl_data._queue);
					assert(error == 0);

					weights = builder.CreateVector<float>(layer_weights, num_weights);
					delta_weights = builder.CreateVector<float>(layer_delta_weights, num_weights);

					delete[] layer_weights;
					delete[] layer_delta_weights;
				}
				else
				{
					num_weights = layer._weights_data->_num_weights;
					
					weights = builder.CreateVector<float>(layer._weights_data->_weights, num_weights);
					delta_weights = builder.CreateVector<float>(layer._weights_data->_delta_weights, num_weights);
				}
		
		
				layers.push_back(Serialization::CreateLayer(builder, weights, delta_weights));
			}
		
			auto serialized_net = builder.CreateVector(layers);
		
			builder.Finish(Serialization::CreateNetData(builder, serialized_net));
			return flatbuffers::SaveFile(path, (char*)builder.GetBufferPointer(), builder.GetSize(), true);
		}
		
		bool LoadNet(NeuralNet& net, const char* path)
		{
			std::ifstream stream(path, std::ifstream::binary);
			std::string data(std::istreambuf_iterator<char>(stream), (std::istreambuf_iterator<char>()));
		
			if (!data[0]) 
			{
				std::cout << "Can't load data, LoadNet()" << std::endl;
				return false;
			}
		
			const Serialization::NetData* net_data = Serialization::GetNetData(data.data());
		
			for (unsigned i = 0; i < net_data->_layer()->size(); i++)
			{
				LayerData& layer = net._layers[i];
		
				//Output layer have no weights
				if (!layer._weights_data && !layer._weights_CL_data)
					continue;
		
				const Serialization::Layer* data_layer = net_data->_layer()->Get(i);
		
				if (layer._use_open_CL)
				{
					const int _size = sizeof(float) * layer._weights_CL_data->_num_weights;
					
					cl_int error = 0;

					error = clFinish(net._cl_data._queue);
					assert(error == 0);

					error = clEnqueueWriteBuffer(net._cl_data._queue, layer._weights_CL_data->_buffer_weights, CL_FALSE, 0, _size, data_layer->_weight()->Data(), NULL, NULL, NULL);
					assert(error == 0);
					error = clEnqueueWriteBuffer(net._cl_data._queue, layer._weights_CL_data->_buffer_delta_weights, CL_FALSE, 0, _size, data_layer->_delta_weight()->Data(), NULL, NULL, NULL);
					assert(error == 0);

					error = clFinish(net._cl_data._queue);
					assert(error == 0);
				}
				else
				{
					const int _size = sizeof(float) * layer._weights_data->_num_weights;

					memcpy(layer._weights_data->_weights, data_layer->_weight()->Data(), _size);
					memcpy(layer._weights_data->_delta_weights, data_layer->_delta_weight()->Data(), _size);
				}
			}
		
			return true;
		}
		
		void SaveTrainingData(Net_ArrayTrainingData training_data, const char* path)
		{
			flatbuffers::FlatBufferBuilder builder(1024);
		
			std::vector<flatbuffers::Offset<Serialization::Data>> _data;
		
			for (size_t i = 0; i < training_data._size; i++)
			{
				const Net_TrainingData time_steep = training_data._data[i];
		
				flatbuffers::Offset<flatbuffers::Vector<float>> input = builder.CreateVector(time_steep._input._data, time_steep._input._size);
				flatbuffers::Offset<flatbuffers::Vector<float>> output = builder.CreateVector(time_steep._output._data, time_steep._output._size);
		
				_data.push_back(Serialization::CreateData(builder, input, output));
			}
		
			auto serialized_training_data = builder.CreateVector(_data);
		
			builder.Finish(Serialization::CreateTrainingData(builder, serialized_training_data));
			flatbuffers::SaveFile(path, (char*)builder.GetBufferPointer(), builder.GetSize(), true);
		}
		
		Net_ArrayTrainingData LoadTrainingData(const char* path)//TODO: Maybe pass training _data as ref and return bool true if successful?
		{
			Net_ArrayTrainingData training_data;
			Net_CreateArrayTD(&training_data, 0, 0);
		
			std::ifstream stream(path, std::ifstream::binary);
			std::string src(std::istreambuf_iterator<char>(stream), (std::istreambuf_iterator<char>()));
		
			if (!src[0])
			{
				std::cout << "Can't load data, LoadTrainingData()" << std::endl;
				return training_data;
			}
		
			const Serialization::TrainingData* serialized_training_data = Serialization::GetTrainingData(src.data());
		
			for (unsigned i = 0; i < serialized_training_data->_time_steep()->size(); i++)
			{
				Net_TrainingData data;
				const Serialization::Data* time_steep = serialized_training_data->_time_steep()->Get(i);
				const int input_size = sizeof(float) * time_steep->_input()->size();
				const int output_size = sizeof(float) * time_steep->_output()->size();
		
				Net_CreateArrayF(&data._input, time_steep->_input()->size(), 0);
				Net_CreateArrayF(&data._output, time_steep->_output()->size(), 0);
				Net_MemCpyArrayF(&data._input, (float*)time_steep->_input()->Data(), 0, time_steep->_input()->size());
				Net_MemCpyArrayF(&data._output, (float*)time_steep->_output()->Data(), 0, time_steep->_input()->size());
		
				Net_AddArrayTD(&training_data, data);
			}
		
			return training_data;
		}
		
		void TrainNet(NeuralNet& net, const Net_ArrayTrainingData& training_data, float eta, float momentum, int display_info_rate)
		{
			int display_info_in = display_info_rate - 1;

			float loss = 0.0f;

			for (size_t i = 0; i < training_data._size; i++)
			{
				FeedForward(net, training_data._data[i]._input);
				BackpropUseCurrentState(net, training_data._data[i]._output, eta, momentum);

				loss += CalcLoss(net, training_data._data[i]._output);

				if (display_info_in <= 0 && display_info_rate != 0)
				{
					std::cout << "Num backprop: " << i + 1 << std::endl;
					std::cout << "Loss: " << loss / display_info_rate << std::endl;
					loss = 0.0f;
					display_info_in = display_info_rate - 1;
				}
				else
				{
					display_info_in--;
				}
			}
		}
	}
}