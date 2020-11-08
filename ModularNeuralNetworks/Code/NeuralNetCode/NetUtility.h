#pragma once

#include "NetTypes.h"

#include <vector>
#include <random>
#include <cmath>

namespace Net 
{
	namespace Utility 
	{

		static std::random_device random_device;
		static std::mt19937 random_engine(random_device());
		static std::uniform_real_distribution<float> distribution_one_to_zero(0.0f, 1.0f);

		void ShuffleTrainingData(Types::TrainingDataVector* training_data);

		float GetDerivativ(Types::ActivationFunction function, float value);

		//TODO: do we need vector
		std::vector<float> RescaleImage(std::vector<float> image_data, int width, int height, int new_width, int new_height, bool has_color);

		template<typename T>
		static T RandomizeFloat(T min, T max);

		template<typename T>
		static T RandomizeInt(T min, T max);

		//Inlined functions
		inline float RandomizeOneToZero() {
			return  distribution_one_to_zero(random_engine);
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