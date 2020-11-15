#pragma once

typedef struct ArrayF 
{
	float* data;
	unsigned size;
	unsigned capacity;
} ArrayF;

void CreateArrayF(ArrayF* array_f, unsigned capacity);
void ResizeArrayF(ArrayF* array_f, unsigned new_capacity);
void SetSizeArrayF(ArrayF* array_f, unsigned new_size);
void FreeArrayF(ArrayF* array_f);

void AddArrayF(ArrayF* array_f, float value);
void RemoveArrayF(ArrayF* array_f);
void AddAtIndexArrayF(ArrayF* array_f, unsigned index);
void RemoveAtIndexArrayF(ArrayF* array_f, unsigned index);
