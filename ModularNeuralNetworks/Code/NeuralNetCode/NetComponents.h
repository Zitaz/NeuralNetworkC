#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "NetTypes.h"

#include <stdbool.h>
#include <CL/cl.h>

typedef struct Net_CompNeurons
{
	float* _values;
	float* _gradient;

	unsigned _num_neurons;
	unsigned _num_neurons_next;
} Net_CompNeurons;

typedef struct Net_CompNeuronsCL
{
	cl_mem _buffer_values;
	cl_mem _buffer_gradient;

	unsigned _num_neurons;
	unsigned _num_neurons_next;
} Net_CompNeuronsCL;

typedef struct Net_CompWeights
{
	float* _weights;
	float* _delta_weights;

	unsigned _num_weights;
} Net_CompWeights;

typedef struct Net_CompWeightsCL
{
	cl_mem _buffer_weights;
	cl_mem _buffer_delta_weights;

	unsigned _num_weights;
} Net_CompWeightsCL;

typedef struct Net_CompFCLayer
{
	unsigned _num_neurons_with_bias;
} Net_CompFCLayer;

typedef struct Net_CompConvLayer
{
	unsigned _depth;
	unsigned _length;
	unsigned _filter_length;
	unsigned _num_filters;
	unsigned _stride;
} Net_CompConvLayer;

typedef struct Net_CompOutLayerCL
{
	cl_mem _target_value;
} Net_CompOutLayerCL;

typedef struct Net_LayerData
{
	Net_NetType _type;
	Net_ActivationFunction _function;

	bool _use_open_CL;

	Net_CompNeurons* _neurons_data;
	Net_CompNeuronsCL* _neurons_CL_data;
	Net_CompWeights* _weights_data;
	Net_CompWeightsCL* _weights_CL_data;
	Net_CompFCLayer* _fC_layer_data;
	Net_CompConvLayer* _conv_layer_data;
	Net_CompOutLayerCL* _output_layer_CL_data;
} Net_LayerData;