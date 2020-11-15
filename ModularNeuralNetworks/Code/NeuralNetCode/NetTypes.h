#pragma once

typedef struct Net_ArrayF
{
	float* _data;
	unsigned _size;
	unsigned _capacity;
} Net_ArrayF;

typedef struct Net_TrainingData
{
	Net_ArrayF _input;
	Net_ArrayF _output;
} TrainingData;

typedef struct Net_ArrayTrainingData
{
	TrainingData* _data;
	unsigned _size;
	unsigned _capacity;
} Net_ArrayTrainingData;

typedef enum 
{
	NET_ACTIVATION_FUNC_LINEAR,
	NET_ACTIVATION_FUNC_SIGMOID,
	NET_ACTIVATION_FUNC_LEAKY_RELU,
	NET_ACTIVATION_FUNC_NUM_FUNCTIONS
} Net_ActivationFunction;

typedef enum
{
	NET_TYPE_OUTPUT,
	NET_TYPE_FULLY_CONNECTED,
	NET_TYPE_CONVOLUTIONAL,
} Net_NetType;

void Net_CreateArrayF(Net_ArrayF* array_f, unsigned _capacity, unsigned _size);
void Net_CreateArrayTD(Net_ArrayTrainingData* array_td, unsigned _capacity, unsigned _size);

void Net_SetCapacityArrayF(Net_ArrayF* array_f, unsigned new_capacity);
void Net_SetCapacityArrayTD(Net_ArrayTrainingData* array_td, unsigned new_capacity);

void Net_SetSizeArrayF(Net_ArrayF* array_f, unsigned new_size);
void Net_SetSizeArrayTD(Net_ArrayTrainingData* array_td, unsigned new_size);

void Net_FreeArrayF(Net_ArrayF* array_f);
void Net_FreeArrayTD(Net_ArrayTrainingData* array_td);

void Net_AddArrayF(Net_ArrayF* array_f, float value);
void Net_AddArrayTD(Net_ArrayTrainingData* array_td, TrainingData value);

void Net_RemoveArrayF(Net_ArrayF* array_f);
void Net_RemoveArrayTD(Net_ArrayTrainingData* array_td);

void Net_MemCpyArrayF(Net_ArrayF* array_f, float* src, unsigned to_index, unsigned num_float);
void Net_MemCpyArrayTD(Net_ArrayTrainingData* array_td, TrainingData* src, unsigned to_index, unsigned num_float);

void Net_AddAtIndexArrayF(Net_ArrayF* array_f, unsigned index);
void Net_AddAtIndexArrayTD(Net_ArrayTrainingData* array_td, unsigned index);

void Net_RemoveAtIndexArrayF(Net_ArrayF* array_f, unsigned index);
void Net_RemoveAtIndexArrayTD(Net_ArrayTrainingData* array_td, unsigned index);

void Net_ZeroArrayF(Net_ArrayF* array_f);