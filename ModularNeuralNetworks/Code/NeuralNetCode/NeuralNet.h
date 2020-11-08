#pragma once

#include "NetUtility.h"
#include "NetComponents.h"

#include <vector>

namespace Net
{
	class NeuralNet
	{
	public:

		NeuralNet(unsigned num_layers);
		~NeuralNet();

		unsigned _num_layers;
		unsigned _num_init_layers;

		LayerData* _layers;
	};

	namespace InitData 
	{
		struct NetConvInitData
		{
			Types::ActivationFunction _function;

			unsigned _length;
			unsigned _depth;
			unsigned _filter_length;
			unsigned _num_filters;
			unsigned _stride;
		};

		struct NetFCInitData {
			Types::ActivationFunction _function;

			unsigned _num_neurons;
			unsigned _num_neurons_next;
		};

		struct NetOutputInitData 
		{
			Types::ActivationFunction _function;

			unsigned _num_neurons;
		};
	}

	namespace NetFunc
	{
		void AddConvLayer(NeuralNet& net, InitData::NetConvInitData& data, bool use_open_cl);
		void AddFCLayer(NeuralNet& net, InitData::NetFCInitData& data, bool use_open_cl);
		void AddOutputLayer(NeuralNet& net, InitData::NetOutputInitData& data, bool use_open_cl);

		void BackpropUseCurrentState(NeuralNet& net, const Types::LayerValues& target_output, float eta, float momentum);

		float CalcLoss(NeuralNet& net, const Types::LayerValues& target_output);

		void FeedForward(NeuralNet& net, Types::LayerValues input);

		Types::LayerValues GetOutputValues(NeuralNet& net);

		//void ValidateNet(NeuralNet& net);
		
		bool SaveNet(const NeuralNet& net, const char* path);
		
		bool LoadNet(NeuralNet& net, const char* path);
		
		void SaveTrainingData(std::vector<Types::TrainingData> training_data, const char* path);//TODO: Move to utillity?
		
		Types::TrainingDataVector LoadTrainingData(const char* path);//TODO: Move to utillity?

		void TrainNet(NeuralNet& net, const Types::TrainingDataVector& training_data, float eta, float momentum, int display_info_rate);
	}
}