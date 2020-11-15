#include "NetArrayFunc.h"
//
//#include <assert.h>
//
//void Net_CreateArrayF(Net_ArrayF* array_f, unsigned _capacity, unsigned _size)
//{
//	assert(_size <= _capacity);
//
//	array_f->_data = malloc(_capacity * sizeof(float));
//	array_f->_size = 0;
//	array_f->_capacity = _capacity;
//}
//
//void Net_CreateArrayTD(Net_ArrayTrainingData* array_td, unsigned _capacity, unsigned _size)
//{
//	assert(_size <= _capacity);
//
//	array_td->_data = malloc(_capacity * sizeof(TrainingData));
//	array_td->_size = 0;
//	array_td->_capacity = _capacity;
//}
//
//void Net_SetCapacityArrayF(Net_ArrayF* array_f, unsigned new_capacity)
//{
//	if (new_capacity < array_f->_size)
//		array_f->_size = new_capacity;
//
//	array_f->_capacity = new_capacity;
//	array_f->_data = realloc(array_f->_data, new_capacity * sizeof(float));
//}
//
//void Net_SetCapacityArrayTD(Net_ArrayTrainingData* array_td, unsigned new_capacity)
//{
//	if (new_capacity < array_td->_size) 
//	{
//		for (size_t i = array_td->_size; i < new_capacity; i++)
//		{
//			Net_FreeArrayF(array_td->_data[i]._input._data);
//			Net_FreeArrayF(array_td->_data[i]._output._data);
//		}
//		
//		array_td->_size = new_capacity;
//	}
//
//	array_td->_capacity = new_capacity;
//	array_td->_data = realloc(array_td->_data, new_capacity * sizeof(TrainingData));
//}
//
//void Net_SetSizeArrayF(Net_ArrayF* array_f, unsigned new_size)
//{
//	if (new_size > array_f->_capacity)
//		Net_SetCapacityArrayF(array_f, new_size);
//
//	array_f->_size = new_size;
//}
//
//void Net_SetSizeArrayTD(Net_ArrayTrainingData* array_td, unsigned new_size)
//{
//	if (new_size > array_td->_capacity)
//		Net_SetCapacityArrayTD(array_td, new_size);
//
//	array_td->_size = new_size;
//}
//
//void Net_FreeArrayF(Net_ArrayF* array_f)
//{
//	free(array_f->_data);
//}
//
//void Net_FreeArrayTD(Net_ArrayTrainingData* array_td)
//{
//	for (size_t i = 0; i < array_td->_size; i++)
//	{
//		Net_FreeArrayF(array_td->_data[i]._input._data);
//		Net_FreeArrayF(array_td->_data[i]._output._data);
//	}
//	free(array_td->_data);
//}
//
//void Net_AddArrayF(Net_ArrayF* array_f, float value)
//{
//	if (array_f->_size == array_f->_capacity) 
//		Net_SetCapacityArrayF(array_f, array_f->_capacity * 2);
//
//	array_f->_data[array_f->_size++] = value;
//}
//
//void Net_AddArrayTD(Net_ArrayTrainingData* array_td, TrainingData value)
//{
//	if (array_td->_size == array_td->_capacity)
//		Net_SetCapacityArrayTD(array_td, array_td->_capacity * 2);
//
//	array_td->_data[array_td->_size++] = value;
//}
//
//void Net_RemoveArrayF(Net_ArrayF* array_f)
//{
//	array_f->_size--;
//}
//
//void Net_RemoveArrayTD(Net_ArrayTrainingData* array_td)
//{
//	array_td->_size--;
//}
//
//void Net_MemCpyArrayF(Net_ArrayF* array_f, float* src, unsigned to_index, unsigned num_float)
//{
//	if (num_float + to_index <= array_f->_capacity)
//		Net_SetCapacityArrayF(array_f, num_float + to_index);
//
//	memcpy(&array_f->_data[to_index], src, num_float * sizeof(float));
//	array_f->_size = to_index + num_float;
//}
//
//void Net_MemCpyArrayTD(Net_ArrayTrainingData* array_td, TrainingData* src, unsigned to_index, unsigned num_float)
//{
//	if (num_float + to_index <= array_td->_capacity)
//		Net_SetCapacityArrayTD(array_td, num_float + to_index);
//
//	memcpy(&array_td->_data[to_index], src, num_float * sizeof(TrainingData));
//	array_td->_size = to_index + num_float;
//}
//
//void Net_AddAtIndexArrayF(Net_ArrayF* array_f, unsigned index)
//{
//	assert(index >= 0);
//	assert(index < array_f->_size);
//
//	memcpy(&array_f->_data[index + 1], &array_f->_data[index], (array_f->_size - index) * sizeof(float));
//	array_f->_size++;
//}
//
//void Net_AddAtIndexArrayTD(Net_ArrayTrainingData* array_td, unsigned index)
//{
//	assert(index >= 0);
//	assert(index < array_td->_size);
//
//	memcpy(&array_td->_data[index + 1], &array_td->_data[index], (array_td->_size - index) * sizeof(TrainingData));
//	array_td->_size++;
//}
//
//void Net_RemoveAtIndexArrayF(Net_ArrayF* array_f, unsigned index)
//{
//	assert(index >= 0);
//	assert(index < array_f->_size);
//
//	array_f->_size--;
//	memcpy(&array_f->_data[index], &array_f->_data[index + 1], (array_f->_size - index) * sizeof(float));
//}
//
//void Net_RemoveAtIndexArrayTD(Net_ArrayTrainingData* array_td, unsigned index)
//{
//	assert(index >= 0);
//	assert(index < array_td->_size);
//
//	array_td->_size--;
//	memcpy(&array_td->_data[index], &array_td->_data[index + 1], (array_td->_size - index) * sizeof(float));
//}
//