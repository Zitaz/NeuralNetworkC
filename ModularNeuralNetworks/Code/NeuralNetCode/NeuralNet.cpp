#include "NeuralNet.h"
#include "Layers/NetConvFunc.h"
#include "Layers/NetOutputFunc.h"
#include "Layers/NetFCFunc.h"
#include "NetBaseFunc.h"
#include "OpenCLFunctions.h"
#include "NetUtility.h"

#include "flatbuffers.h"
#include "Generated/NetData_generated.h"
#include "Generated/TrainingData_generated.h"
#include "util.h"

#include <assert.h>
#include <math.h>
#include <iostream>

namespace Net
{
	NeuralNet::NeuralNet(unsigned num_layers)
	{
		_num_layers = num_layers;
		_num_init_layers = 0;
		_layers = new LayerData[num_layers];
	}

	NeuralNet::~NeuralNet()
	{
		if (_layers)
			delete[] _layers;
	}

	namespace NetFunc
	{
		void AddConvLayer(NeuralNet& net, InitData::NetConvInitData& data, bool use_open_cl)//TODO: Can probably split into more function for reuse
		{
			assert(net._num_init_layers < net._num_layers, "Can't add more layers");

			LayerData& layer = net._layers[net._num_init_layers];

			const unsigned num_neurons = data._length * data._length * data._depth;//TODO: Do we need a bias in conv layer?
			const unsigned num_neurons_next = ((data._length - data._filter_length) / data._stride + 1) * ((data._length - data._filter_length) / data._stride + 1) * data._num_filters;
			const unsigned num_weights = data._filter_length * data._filter_length * data._depth * data._num_filters;

			layer._conv_layer_data = new ConvLayerData;

			layer._type = Types::NetType::CONVOLUTIONAL;
			layer._function = data._function;
			layer._use_open_CL = use_open_cl;

			layer._conv_layer_data->_depth = data._depth;
			layer._conv_layer_data->_filter_length = data._filter_length;
			layer._conv_layer_data->_length = data._length;
			layer._conv_layer_data->_num_filters = data._num_filters;
			layer._conv_layer_data->_stride = data._stride;

			float* weights = new float[num_weights];

			for (size_t i = 0; i < num_weights; i++)
			{
				//Todo: fix so weight init is correct for every activation function. It's currently using the best known for relu
				weights[i] = Utility::RandomizeOneToZero() * (2.0 / (data._filter_length * data._filter_length * data._depth));
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
				layer._neurons_CL_data->_buffer_values = clCreateBuffer(OpenCL::Data::instace.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_neurons, values, &error);
				assert(error == 0);
				layer._neurons_CL_data->_buffer_gradient = clCreateBuffer(OpenCL::Data::instace.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_neurons, values, &error);
				assert(error == 0);
				layer._weights_CL_data->_buffer_weights = clCreateBuffer(OpenCL::Data::instace.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_weights, weights, &error);
				assert(error == 0);
				layer._weights_CL_data->_buffer_delta_weights = clCreateBuffer(OpenCL::Data::instace.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_weights, delta_weights, &error);
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

		void AddFCLayer(NeuralNet& net, InitData::NetFCInitData& data, bool use_open_cl)
		{
			assert(net._num_init_layers < net._num_layers, "Can't add more layers");

			LayerData& layer = net._layers[net._num_init_layers];

			const unsigned num_neurons_with_bias = data._num_neurons + 1;
			const unsigned num_weights = num_neurons_with_bias * data._num_neurons_next;

			layer._fC_layer_data = new FCLayerData;

			layer._type = Types::NetType::FULLY_CONNECTED;
			layer._function = data._function;
			layer._use_open_CL = use_open_cl;

			layer._fC_layer_data->_num_neurons_with_bias = num_neurons_with_bias;

			float* weights = new float[num_weights];

			for (size_t i = 0; i < num_weights; i++)
			{
				//Todo: fix so weight init is correct for every activation function. It's currently using the best known for relu
				weights[i] = Utility::RandomizeOneToZero() * (2.0 / num_neurons_with_bias);
			}

			if (use_open_cl)
			{
				layer._neurons_CL_data = new NeuronsCLData;
				layer._weights_CL_data = new WeightsCLData;

				layer._neurons_CL_data->_num_neurons = data._num_neurons;
				layer._neurons_CL_data->_num_neurons_next = data._num_neurons_next;

				layer._weights_CL_data->_num_weights = num_weights;

				float* values = new float[num_neurons_with_bias] { 0 };
				float* delta_weights = new float[num_weights] { 0 };

				values[num_neurons_with_bias - 1] = 1.0f;//Set bias

				cl_int error = 0;
				layer._neurons_CL_data->_buffer_values = clCreateBuffer(OpenCL::Data::instace.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_neurons_with_bias, values, &error);
				assert(error == 0);
				layer._neurons_CL_data->_buffer_gradient = clCreateBuffer(OpenCL::Data::instace.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_neurons_with_bias, values, &error);
				assert(error == 0);
				layer._weights_CL_data->_buffer_weights = clCreateBuffer(OpenCL::Data::instace.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_weights, weights, &error);
				assert(error == 0);
				layer._weights_CL_data->_buffer_delta_weights = clCreateBuffer(OpenCL::Data::instace.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_weights, delta_weights, &error);
				assert(error == 0);

				delete[] weights;
				delete[] values;
				delete[] delta_weights;
			}
			else
			{
				layer._neurons_data = new NeuronsData;
				layer._weights_data = new WeightsData;

				layer._neurons_data->_num_neurons = data._num_neurons;
				layer._neurons_data->_num_neurons_next = data._num_neurons_next;

				layer._weights_data->_num_weights = num_weights;

				layer._neurons_data->_values = new float[num_neurons_with_bias] { 0 };
				layer._neurons_data->_gradient = new float[num_neurons_with_bias] { 0 };//TODO: Do we need gradient for bias?
				layer._weights_data->_delta_weights = new float[num_weights] { 0 };
				layer._weights_data->_weights = weights;//Takes ownership

				layer._neurons_data->_values[num_neurons_with_bias - 1] = 1.0f;//Set bias
			}

			net._num_init_layers++;
		}

		void AddOutputLayer(NeuralNet& net, InitData::NetOutputInitData& data, bool use_open_cl)
		{
			assert(net._num_init_layers < net._num_layers, "Can't add more layers");

			LayerData& layer = net._layers[net._num_init_layers];

			layer._type = Types::NetType::OUTPUT;
			layer._function = data._function;
			layer._use_open_CL = use_open_cl;

			if (use_open_cl)
			{
				layer._neurons_CL_data = new NeuronsCLData;
				layer._output_layer_CL_data = new OutputLayerCLData;

				layer._neurons_CL_data->_num_neurons = data._num_neurons;

				float* values = new float[data._num_neurons]{ 0 };//TODO: Maybe there is a beter way to initilize valus to zero in the buffers?

				cl_int error = 0;
				layer._neurons_CL_data->_buffer_values = clCreateBuffer(OpenCL::Data::instace.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * data._num_neurons, values, &error);
				assert(error == 0);
				layer._neurons_CL_data->_buffer_gradient = clCreateBuffer(OpenCL::Data::instace.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * data._num_neurons, values, &error);
				assert(error == 0);
				layer._output_layer_CL_data->_target_value = clCreateBuffer(OpenCL::Data::instace.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * data._num_neurons, values, &error);
				assert(error == 0);

				delete[] values;
			}
			else
			{
				layer._neurons_data = new NeuronsData;

				layer._neurons_data->_num_neurons = data._num_neurons;

				layer._neurons_data->_values = new float[data._num_neurons]{ 0 };
				layer._neurons_data->_gradient = new float[data._num_neurons]{ 0 };
			}

			net._num_init_layers++;
		}

		void BackpropUseCurrentState(NeuralNet& net, const Types::LayerValues& target_output, float eta, float momentum)
		{
			for (int i = net._num_layers - 1; i >= 0; i--)
			{
				LayerData& layer = net._layers[i];
				LayerData& next_layer = net._layers[i + 1];

				switch (layer._type)
				{
				case Types::NetType::FULLY_CONNECTED:
					FCFunc::Backprop(layer, next_layer, eta, momentum);
					break;
				case Types::NetType::CONVOLUTIONAL:
					ConvFunc::Backprop(layer, next_layer, eta, momentum);
					break;
				case Types::NetType::OUTPUT:
					OutputFunc::Backprop(layer, target_output);
					break;
				}
			}
		}

		float CalcLoss(NeuralNet& net, const Types::LayerValues& target_output)
		{
			return OutputFunc::GetLoss(net._layers[net._num_layers - 1], target_output);
		}

		void FeedForward(NeuralNet& net, Types::LayerValues input)
		{
			//Set values of the input layer
			BaseFunc::SetValues(net._layers[0], input);

			for (unsigned i = 0; i < net._num_layers - 1; i++)
			{
				LayerData& layer = net._layers[i];
				LayerData& next_layer = net._layers[i + 1];

				switch (layer._type)
				{
				case Types::NetType::FULLY_CONNECTED:
					FCFunc::FeedForward(layer, next_layer);
					break;
				case Types::NetType::CONVOLUTIONAL:
					ConvFunc::FeedForward(layer, next_layer);
					break;
				}
			}
		}

		Types::LayerValues GetOutputValues(NeuralNet& net)
		{
			return BaseFunc::GetValues(net._layers[net._num_layers - 1]);
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
		//		if (layer._type == BaseLayer::CONVOLUTIONAL)
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
		//		if (layer._type == NetLayer::NetType::FULLY_CONNECTED) {
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
		
					error = clFinish(OpenCL::Data::instace.queue);
					assert(error == 0);

					error = clEnqueueReadBuffer(OpenCL::Data::instace.queue, layer._weights_CL_data->_buffer_weights, CL_FALSE, 0, sizeof(float) * num_weights, layer_weights, NULL, NULL, NULL);
					assert(error == 0);
					error = clEnqueueReadBuffer(OpenCL::Data::instace.queue, layer._weights_CL_data->_buffer_delta_weights, CL_FALSE, 0, sizeof(float) * num_weights, layer_delta_weights, NULL, NULL, NULL);
					assert(error == 0);

					error = clFinish(OpenCL::Data::instace.queue);
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
					const int size = sizeof(float) * layer._weights_CL_data->_num_weights;
					
					cl_int error = 0;

					error = clFinish(OpenCL::Data::instace.queue);
					assert(error == 0);

					error = clEnqueueWriteBuffer(OpenCL::Data::instace.queue, layer._weights_CL_data->_buffer_weights, CL_FALSE, 0, size, data_layer->_weight()->Data(), NULL, NULL, NULL);
					assert(error == 0);
					error = clEnqueueWriteBuffer(OpenCL::Data::instace.queue, layer._weights_CL_data->_buffer_delta_weights, CL_FALSE, 0, size, data_layer->_delta_weight()->Data(), NULL, NULL, NULL);
					assert(error == 0);

					error = clFinish(OpenCL::Data::instace.queue);
					assert(error == 0);
				}
				else
				{
					const int size = sizeof(float) * layer._weights_data->_num_weights;

					memcpy(layer._weights_data->_weights, data_layer->_weight()->Data(), size);
					memcpy(layer._weights_data->_delta_weights, data_layer->_delta_weight()->Data(), size);
				}
			}
		
			return true;
		}
		
		void SaveTrainingData(std::vector<Types::TrainingData> training_data, const char* path)
		{
			flatbuffers::FlatBufferBuilder builder(1024);
		
			std::vector<flatbuffers::Offset<Serialization::Data>> data;
		
			for (size_t i = 0; i < training_data.size(); i++)
			{
				const Types::TrainingData time_steep = training_data[i];
		
				flatbuffers::Offset<flatbuffers::Vector<float>> input = builder.CreateVector(time_steep._input);
				flatbuffers::Offset<flatbuffers::Vector<float>> output = builder.CreateVector(time_steep._output);
		
				data.push_back(Serialization::CreateData(builder, input, output));
			}
		
			auto serialized_training_data = builder.CreateVector(data);
		
			builder.Finish(Serialization::CreateTrainingData(builder, serialized_training_data));
			flatbuffers::SaveFile(path, (char*)builder.GetBufferPointer(), builder.GetSize(), true);
		}
		
		Types::TrainingDataVector LoadTrainingData(const char * path)//TODO: Maybe pass training data as ref and return bool true if successful?
		{
			std::vector<Types::TrainingData> training_data;
		
			std::ifstream stream(path, std::ifstream::binary);
			std::string data(std::istreambuf_iterator<char>(stream), (std::istreambuf_iterator<char>()));
		
			if (!data[0])
			{
				std::cout << "Can't load data, LoadTrainingData()" << std::endl;
				return training_data;
			}
		
			const Serialization::TrainingData* serialized_training_data = Serialization::GetTrainingData(data.data());
		

			for (unsigned i = 0; i < serialized_training_data->_time_steep()->size(); i++)
			{
				Types::TrainingData data;
				const Serialization::Data* time_steep = serialized_training_data->_time_steep()->Get(i);
				const int input_size = sizeof(float) * time_steep->_input()->size();
				const int output_size = sizeof(float) * time_steep->_output()->size();
		
				data._input.resize(time_steep->_input()->size());
				data._output.resize(time_steep->_output()->size());
		
				memcpy(data._input.data(), time_steep->_input()->Data(), input_size);
				memcpy(data._output.data(), time_steep->_output()->Data(), output_size);
		
				training_data.push_back(data);
			}
		
			return training_data;
		}
		
		void TrainNet(NeuralNet& net, const Types::TrainingDataVector& training_data, float eta, float momentum, int display_info_rate)
		{
			int display_info_in = display_info_rate - 1;

			float loss = 0.0f;

			for (size_t i = 0; i < training_data.size(); i++)
			{
				FeedForward(net, training_data[i]._input);
				BackpropUseCurrentState(net, training_data[i]._output, eta, momentum);

				loss += CalcLoss(net, training_data[i]._output);

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