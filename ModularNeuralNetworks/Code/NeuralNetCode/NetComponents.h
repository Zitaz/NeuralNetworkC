#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

extern "C" {
#include "NetTypes.h"
}

#include <CL/cl.h>

namespace Net 
{
	typedef struct NeuronsData
	{
		float* _values = nullptr;
		float* _gradient = nullptr;

		unsigned _num_neurons;
		unsigned _num_neurons_next;

		~NeuronsData() {
			if (_values)
				delete[] _values;
			if (_gradient)
				delete[] _gradient;
		}
	} NeuronsData;

	typedef struct NeuronsCLData
	{
		cl_mem _buffer_values;
		cl_mem _buffer_gradient;

		unsigned _num_neurons;
		unsigned _num_neurons_next;
	} NeuronsCLData;

	typedef struct WeightsData
	{
		float* _weights;
		float* _delta_weights;

		unsigned _num_weights;

		~WeightsData() {
			if (_weights)
				delete[] _weights;
			if (_delta_weights)
				delete[] _delta_weights;
		}
	} WeightsData;

	typedef struct WeightsCLData
	{
		cl_mem _buffer_weights;//TODO: delete clReleaseMemObject
		cl_mem _buffer_delta_weights;

		unsigned _num_weights;
	} WeightsCLData;

	typedef struct FCLayerData
	{
		unsigned _num_neurons_with_bias;
	} FCLayerData;

	typedef struct ConvLayerData
	{
		unsigned _depth;
		unsigned _length;
		unsigned _filter_length;
		unsigned _num_filters;
		unsigned _stride;
	} ConvLayerData;

	typedef struct OutputLayerCLData
	{
		cl_mem _target_value;
	} OutputLayerCLData;

	typedef struct LayerData
	{
		Net_NetType _type;
		Net_ActivationFunction _function;

		bool _use_open_CL;

		NeuronsData* _neurons_data = nullptr;
		NeuronsCLData* _neurons_CL_data = nullptr;
		WeightsData* _weights_data = nullptr;
		WeightsCLData* _weights_CL_data = nullptr;
		FCLayerData* _fC_layer_data = nullptr;
		ConvLayerData* _conv_layer_data = nullptr;
		OutputLayerCLData* _output_layer_CL_data = nullptr;

		~LayerData() 
		{
			if (_neurons_data) 
				delete _neurons_data;
			if (_neurons_CL_data)
				delete _neurons_CL_data;
			if (_weights_data)
				delete _weights_data;
			if (_weights_CL_data)
				delete _weights_CL_data;
			if (_fC_layer_data)
				delete _fC_layer_data;
			if (_conv_layer_data)
				delete _conv_layer_data;
			if (_output_layer_CL_data)
				delete _output_layer_CL_data;
		}
	} LayerData;
}