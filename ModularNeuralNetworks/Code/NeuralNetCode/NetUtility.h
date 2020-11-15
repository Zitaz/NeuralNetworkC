#pragma once

#include "NetTypes.h"
#include "ArrayF.h"

#include <stdio.h>
#include <cmath>
#include <time.h> 

namespace Net 
{
	namespace Utility 
	{
		static bool randInit = false;

		//TODO: Convert to c
		//void ShuffleTrainingData(Types::TrainingDataVector* training_data);

		float GetDerivativ(Types::ActivationFunction function, float value);

		//TODO: convert to c
		//std::vector<float> RescaleImage(std::vector<float> image_data, int width, int height, int new_width, int new_height, bool has_color);

		//Inlined functions
		inline float RandomizeOneToZero() {
			if (randInit == false) {
				srand(time(0));
				randInit = true;
			}

			return rand() % 2;
		}

		inline float Relu(float value)
		{
			if (value > 0.0f)
				return value;
			else
				return value * 0.01f;
		}

		inline float ReluDerivative(float value)
		{
			if (value >= 0.0f)
				return 1.0f;
			else
				return 0.01f;
		}

		inline float Sigmoid(float value)
		{
			return 1 / (1 + exp(-value));
		}

		inline float SigmoidDerivative(float value)
		{
			return Sigmoid(value) * (1 - Sigmoid(value));
		}
	}
}