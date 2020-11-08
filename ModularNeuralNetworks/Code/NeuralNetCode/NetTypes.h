#pragma once

#include <vector>

namespace Net
{
	namespace Types
	{
		typedef std::vector<float> LayerValues;

		struct TrainingData
		{
			LayerValues _input;
			LayerValues _output;
		};

		typedef std::vector<TrainingData> TrainingDataVector;

		enum ActivationFunction
		{
			LINEAR,
			SIGMOID,
			LEAKY_RELU,
			NUM_FUNCTIONS
		};

		enum NetType
		{
			OUTPUT,
			FULLY_CONNECTED,
			CONVOLUTIONAL,
		};
	}
}