#include "ArrayF.h"

#include <assert.h>

void CreateArrayF(ArrayF* array_f, unsigned capacity)
{
	array_f->data = malloc(capacity * sizeof(float));
	array_f->size = 0;
	array_f->capacity = capacity;
}

void ResizeArrayF(ArrayF* array_f, unsigned new_capacity)
{
	if (new_capacity < array_f->size)
		array_f->size = new_capacity;

	array_f->capacity = new_capacity;
	array_f->data = realloc(array_f->data, new_capacity * sizeof(float));
}

void SetSizeArrayF(ArrayF* array_f, unsigned new_size)
{
	array_f->size = new_size;
}

void FreeArrayF(ArrayF* array_f)
{
	free(array_f->data);
}

void AddArrayF(ArrayF* array_f, float value)
{
	if (array_f->size == array_f->capacity) 
		ResizeArrayF(array_f, array_f->capacity * 2);

	array_f->data[array_f->size++] = value;
}

void RemoveArrayF(ArrayF * array_f)
{
	array_f->size--;
}

void AddAtIndexArrayF(ArrayF * array_f, unsigned index)
{
	assert(index >= 0);
	assert(index < array_f->size);

	memcpy(&array_f->data[index + 1], &array_f->data[index], (array_f->size - index) * sizeof(float));
	array_f->size++;
}

void RemoveAtIndexArrayF(ArrayF* array_f, unsigned index)
{
	assert(index >= 0);
	assert(index < array_f->size);

	array_f->size--;
	memcpy(&array_f->data[index], &array_f->data[index + 1], (array_f->size - index) * sizeof(float));
}
