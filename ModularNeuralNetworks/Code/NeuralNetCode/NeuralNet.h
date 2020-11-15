#pragma once

#include "NetUtility.h"
#include "NetComponents.h"

extern "C" {
#include "OpenCLFunctions.h"
}

#include <vector>

namespace Net
{
	typedef struct NeuralNet
	{
	public:

		~NeuralNet();

		unsigned _num_layers;
		unsigned _num_init_layers;

		Net_CLData _cl_data;
		LayerData* _layers;
	} NeuralNet;

	namespace InitData 
	{
		typedef struct NetConvInitData
		{
			Net_ActivationFunction _function;

			unsigned _length;
			unsigned _depth;
			unsigned _filter_length;
			unsigned _num_filters;
			unsigned _stride;
		} NetConvInitData;

		typedef struct NetFCInitData 
		{
			Net_ActivationFunction _function;

			unsigned _num_neurons;
			unsigned _num_neurons_next;
		} NetFCInitData;

		typedef struct NetOutputInitData
		{
			Net_ActivationFunction _function;

			unsigned _num_neurons;
		} NetOutputInitData;
	}

	namespace NetFunc
	{
		void CreateNeuralNet(NeuralNet* net, unsigned num_layers);

		void AddConvLayer(NeuralNet& net, InitData::NetConvInitData& _data, bool use_open_cl);
		void AddFCLayer(NeuralNet& net, InitData::NetFCInitData& _data, bool use_open_cl);
		void AddOutputLayer(NeuralNet& net, InitData::NetOutputInitData& _data, bool use_open_cl);

		void BackpropUseCurrentState(NeuralNet& net, const Net_ArrayF& target_output, float eta, float momentum);

		float CalcLoss(NeuralNet& net, const Net_ArrayF& target_output);

		void FeedForward(NeuralNet& net, Net_ArrayF input);

		Net_ArrayF GetOutputValues(NeuralNet& net);

		//void ValidateNet(NeuralNet& net);
		
		bool SaveNet(const NeuralNet& net, const char* path);
		
		bool LoadNet(NeuralNet& net, const char* path);
		
		void SaveTrainingData(Net_ArrayTrainingData training_data, const char* path);//TODO: Move to utillity?
		
		Net_ArrayTrainingData LoadTrainingData(const char* path);//TODO: Move to utillity?

		void TrainNet(NeuralNet& net, const Net_ArrayTrainingData& training_data, float eta, float momentum, int display_info_rate);
	}
}