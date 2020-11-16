#pragma once

#include "NetTypes.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> 
#include <cstdbool>

static bool rand_init = false;

//TODO: Convert to c
//void ShuffleTrainingData(Types::TrainingDataVector* training_data);

float Net_UtilGetDerivativ(Net_ActivationFunction function, float value);

//TODO: convert to c
//std::vector<float> RescaleImage(std::vector<float> image_data, int width, int height, int new_width, int new_height, bool has_color);

//Inlined functions
inline float Net_UtilRandomizeOneToZero() {
	if (rand_init == false) {
		srand((unsigned)time(0));
		rand_init = true;
	}

	return  (float)rand() / (float)(0x7fff / 1.0f);
}

inline float Net_UtilRelu(float value)
{
	if (value > 0.0f)
		return value;
	else
		return value * 0.01f;
}

inline float Net_UtilReluDerivative(float value)
{
	if (value >= 0.0f)
		return 1.0f;
	else
		return 0.01f;
}

inline float Net_UtilSigmoid(float value)
{
	return 1.0f / (1.0f + exp(-value));
}

inline float Net_UtilSigmoidDerivative(float value)
{
	return Net_UtilSigmoid(value) * (1 - Net_UtilSigmoid(value));
}