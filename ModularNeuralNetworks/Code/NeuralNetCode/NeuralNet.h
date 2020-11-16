#pragma once

#include "NetComponents.h"
#include "NetUtility.h"
#include "OpenCLFunctions.h"

typedef struct Net_NeuralNet
{
	unsigned _num_layers;
	unsigned _num_init_layers;

	Net_CLData _cl_data;//TODO: Change if it dose not work with more then one neural net
	Net_LayerData* _layers;
} Net_NeuralNet;

typedef struct Net_ConvInitData
{
	Net_ActivationFunction _function;

	unsigned _length;
	unsigned _depth;
	unsigned _filter_length;
	unsigned _num_filters;
	unsigned _stride;
} Net_ConvInitData;

typedef struct Net_FCInitData 
{
	Net_ActivationFunction _function;

	unsigned _num_neurons;
	unsigned _num_neurons_next;
} Net_FCInitData;

typedef struct Net_OutputInitData
{
	Net_ActivationFunction _function;

	unsigned _num_neurons;
} Net_OutputInitData;

void Net_CreateNet(Net_NeuralNet* net, unsigned num_layers);
void Net_FreeLayersNet(Net_NeuralNet* net);

void Net_AddConvLayer(Net_NeuralNet* net, Net_ConvInitData* _data, bool use_open_cl);
void Net_AddFCLayer(Net_NeuralNet* net, Net_FCInitData* _data, bool use_open_cl);
void Net_AddOutputLayer(Net_NeuralNet* net, Net_OutputInitData* _data, bool use_open_cl);

void Net_BackpropUseCurrentState(Net_NeuralNet* net, const Net_ArrayF* target_output, float eta, float momentum);
void Net_FeedForward(Net_NeuralNet* net, Net_ArrayF input);
float Net_CalcLoss(Net_NeuralNet* net, const Net_ArrayF* target_output);

Net_ArrayF Net_GetOutputValues(Net_NeuralNet* net);

//void ValidateNet(NeuralNet& net);

bool Net_SaveNet(const Net_NeuralNet* net, const char* path);
bool Net_LoadNet(Net_NeuralNet* net, const char* path);

void Net_SaveTrainingData(Net_ArrayTrainingData* training_data, const char* path);//TODO: Move to utillity?
Net_ArrayTrainingData Net_LoadTrainingData(const char* path);//TODO: Move to utillity?

void Net_TrainNet(Net_NeuralNet* net, const Net_ArrayTrainingData* training_data, float eta, float momentum, int display_info_rate);