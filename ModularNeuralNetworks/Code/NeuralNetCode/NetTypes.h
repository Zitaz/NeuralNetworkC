#pragma once

#include <vector>

namespace Net
{
	namespace Types
	{
		typedef std::vector<float> LayerValues;

		typedef struct TrainingData
		{
			LayerValues _input;
			LayerValues _output;
		} TrainingData;

		typedef std::vector<TrainingData> TrainingDataVector;

		typedef enum
		{
			LINEAR,
			SIGMOID,
			LEAKY_RELU,
			NUM_FUNCTIONS
		} ActivationFunction;

		typedef enum
		{
			OUTPUT,
			FULLY_CONNECTED,
			CONVOLUTIONAL,
		} NetType;
	}
}