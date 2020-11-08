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
float SigmoidDerivative(float value)
{
	return Sigmoid(value) * (1 - Sigmoid(value));
}

inline void atomicAdd_g_f(volatile __global float *addr, float val)
{
    union {
        unsigned int u32;
        float        f32;
    } next, expected, current;
	current.f32    = *addr;
    do {
	   expected.f32 = current.f32;
        next.f32     = expected.f32 + val;
		current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, 
                            expected.u32, next.u32);
    } while( current.u32 != expected.u32 );
}
inline void atomicAdd_l_f(volatile __local float *addr, float val)
{
    union {
        unsigned int u32;
        float        f32;
    } next, expected, current;
	current.f32    = *addr;
    do {
	   expected.f32 = current.f32;
        next.f32     = expected.f32 + val;
		current.u32  = atomic_cmpxchg( (volatile __local unsigned int *)addr, 
                            expected.u32, next.u32);
    } while( current.u32 != expected.u32 );
}

__kernel void FeedForwardFCRelu(__global float* weights, __global float* values_in, __global float* values_out, int num_neurons)
{
	const size_t x_id = get_global_id(0);
	const size_t num_neurons_next = get_global_size(0);

	float value = 0.0f;

	for(int y = 0; y < num_neurons; y++)
	{
		value += weights[y * num_neurons_next + x_id] * values_in[y];
	}
	values_out[x_id] = Relu(value); 
}

__kernel void FeedForwardFCSigmoid(__global float* weights, __global float* values_in, __global float* values_out, int num_neurons)
{
	const size_t x_id = get_global_id(0);
	const size_t num_neurons_next = get_global_size(0);

	float value = 0.0f;

	for(int y = 0; y < num_neurons; y++)
	{
		value += weights[y * num_neurons_next + x_id] * values_in[y];
	}
	values_out[x_id] = Sigmoid(value); 
}

__kernel void FeedForwardFCLinear(__global float* weights, __global float* values_in, __global float* values_out, int num_neurons)
{
	const size_t x_id = get_global_id(0);
	const size_t num_neurons_next = get_global_size(0);

	float value = 0.0f;

	for(int y = 0; y < num_neurons; y++)
	{
		value += weights[y * num_neurons_next + x_id] * values_in[y];
	}
	values_out[x_id] = value; 
}

//__kernel void BackpropFCRelu(__global float* weights, __global float* delta_weights, __global float* gradient,
//						 __global float* next_gradient, __global float* values, __local float* temp_gradient,
//						 float eta, float momentum, int num_neurons)
//{
//	const size_t next_index = get_global_id(0);
//	const size_t num_neurons_next_no_bias = get_global_size(0);
//	const size_t local_id = get_local_id(0);
//	const size_t local_size = get_local_size(0);
//
//	float new_delta_weight;
//
//	if(local_id == 0)
//	{
//		for (size_t from_index = 0; from_index < num_neurons; from_index++)
//		    temp_gradient[from_index] = 0.0f;
//    }
//	
//	//barrier(CLK_LOCAL_MEM_FENCE);
//	//for (size_t from_index = 0; from_index < num_neurons; from_index++)
//	//{
//	//	const size_t array_index = from_index * num_neurons_next_no_bias + next_index;
//    //
//	//	atomicAdd_l_f(&temp_gradient[from_index], weights[array_index] * next_gradient[next_index]);
//    //
//	//	new_delta_weight =
//	//		eta
//	//		* values[from_index]
//	//		* next_gradient[next_index]
//	//		+ momentum
//	//		* delta_weights[array_index];
//	//	
//	//	delta_weights[array_index] = new_delta_weight;
//	//	weights[array_index] += new_delta_weight;
//	//}
//    //
//	//barrier(CLK_LOCAL_MEM_FENCE);
//	//if(local_id == 0)
//	//{
//	//	for (size_t from_index = 0; from_index < num_neurons; from_index++)
//	//	   atomicAdd_g_f(&gradient[from_index], temp_gradient[from_index]);
//    //}
//	//
//	//barrier(CLK_GLOBAL_MEM_FENCE);
//	//if(next_index == 0)
//	//{ 
//	//	for (size_t from_index = 0; from_index < num_neurons; from_index++)
//	//		gradient[from_index] = gradient[from_index] * ReluDerivative(values[from_index]);
//	//}
//}

__kernel void BackpropFCRelu(__global float* weights, __global float* delta_weights, __global float* gradient, __global float* next_gradient, __global float* values,
						 float eta, float momentum, int num_neurons_next_no_bias)
{
	const size_t this_index = get_global_id(0);
	const size_t num_neurons = get_global_size(0);
	const size_t local_id = get_local_id(0);
	const size_t local_size = get_local_size(0);

	float new_delta_weight;
	float new_gradient = 0.0f;
	
	for (size_t next_index = 0; next_index < num_neurons_next_no_bias; next_index++)
	{
		const size_t array_index = this_index * num_neurons_next_no_bias + next_index;
    
		new_gradient += weights[array_index] * next_gradient[next_index];
    
		new_delta_weight =
			eta
			* values[this_index]
			* next_gradient[next_index]
			+ momentum
			* delta_weights[array_index];
		
		delta_weights[array_index] = new_delta_weight;
		weights[array_index] += new_delta_weight;
	}
	
	gradient[this_index] = new_gradient * ReluDerivative(values[this_index]);
}

//__kernel void UpdateWeightsFC(__global float* weights, __global float* delta_weights, __global float* next_gradient, __global float* values,
//						 float eta, float momentum, int num_neurons)
//{
//	const size_t next_index = get_global_id(0);
//	const size_t num_neurons_next_no_bias = get_global_size(0);
//	const size_t local_id = get_local_id(0);
//	const size_t local_size = get_local_size(0);
//
//	float new_delta_weight = 0.0f;
//	
//	for (size_t from_index = 0; from_index < num_neurons; from_index++)
//	{
//		new_delta_weight =
//			eta
//			* values[from_index]
//			* next_gradient[next_index]
//			+ momentum
//			* delta_weights[array_index];
//		
//		delta_weights[array_index] = new_delta_weight;
//		weights[array_index] += new_delta_weight;
//	}
//}

__kernel void BackpropOutputSigmoid(__global float* gradient, __global float* values, __global float* target_output)
{
	const size_t x_id = get_global_id(0);

	float value = values[x_id];
	gradient[x_id] = (target_output[x_id] - value) * SigmoidDerivative(value);
}

__kernel void BackpropOutputLinear(__global float* gradient, __global float* values, __global float* target_output)
{
	const size_t x_id = get_global_id(0);

	gradient[x_id] = (target_output[x_id] - values[x_id]);
}
__kernel void FeedForwardConvRelu(__global float* weights, __global float* values_in, __global float* values_out,
								  int stride, int filter_length, int depth)
{
	const size_t x_id = get_global_id(0);
	const size_t y_id = get_global_id(1);
	const size_t z_id = get_global_id(2);
	const size_t next_length = get_global_size(0);
	const size_t num_filters = get_global_size(2);

	const size_t next_length_pow = next_length * next_length;
	const size_t length = (next_length - 1) * stride + filter_length;
	const size_t length_pow = length * length;
	const size_t filter_length_pow = filter_length * filter_length;
	const size_t num_weights_filter = filter_length_pow * depth;
	//Calculating first to_value index for this filter
	const size_t to_value_z_index = z_id * next_length_pow;
	//Calculating index of first weight for this filter
	const size_t weight_start_index = z_id * num_weights_filter;
	//Calculating to_value index in array using to_z_start_index
	const size_t to_value_y_index = to_value_z_index + y_id * next_length;
	
	float value = 0;

	//It maybe more cash friendly to calculate weight_z here if you have many layers
	for (size_t weight_z = 0; weight_z < depth; ++weight_z)
	{
		//Calculating weight z index in array
		const size_t weight_z_index = weight_start_index + weight_z * filter_length_pow;
		//Calculating z start index in value from array
		const size_t from_z_start_index = weight_z * length_pow;

		for (size_t weight_y = 0; weight_y < filter_length; ++weight_y)
		{
			//Calculating weight z index in array
			const size_t weight_y_index = weight_z_index + weight_y * filter_length;
			//Uses y to calculate y index in value from array
			//Don't need to add padding cus we whant to get the index in the top left corner of kernal
			//Then add weight_y to get to the corect neuron for that weight
			const size_t from_y_start_index = from_z_start_index + ((y_id * stride + weight_y) * length);

			for (size_t weight_x = 0; weight_x < filter_length; ++weight_x)
			{
				value += weights[weight_y_index + weight_x] * values_in[from_y_start_index + weight_x + x_id * stride];
			}
		}
	}
	values_out[to_value_y_index + x_id] = Relu(value); 
}
__kernel void FeedForwardConvLinear(__global float* weights, __global float* values_in, __global float* values_out,
								  int stride, int filter_length, int this_depth)
{
	const size_t x_id = get_global_id(0);
	const size_t y_id = get_global_id(1);
	const size_t z_id = get_global_id(2);
	const size_t next_length = get_global_size(0);
	const size_t num_filters = get_global_size(2);

	const size_t next_length_pow = next_length * next_length;
	const size_t this_length = (next_length - 1) * stride + filter_length;
	const size_t length_pow = this_length * this_length;
	const size_t filter_length_pow = filter_length * filter_length;
	const size_t num_weights_filter = filter_length_pow * this_depth;
	//Calculating first to_value index for this filter
	const size_t to_value_z_index = z_id * next_length_pow;
	//Calculating index of first weight for this filter
	const size_t weight_start_index = z_id * num_weights_filter;
	//Calculating to_value index in array using to_z_start_index
	const size_t to_value_y_index = to_value_z_index + y_id * next_length;
	
	float value = 0;

	//It maybe more cash friendly to calculate weight_z here if you have many layers
	for (size_t weight_z = 0; weight_z < this_depth; ++weight_z)
	{
		//Calculating weight z index in array
		const size_t weight_z_index = weight_start_index + weight_z * filter_length_pow;
		//Calculating z start index in value from array
		const size_t from_z_start_index = weight_z * length_pow;

		for (size_t weight_y = 0; weight_y < filter_length; ++weight_y)
		{
			//Calculating weight z index in array
			const size_t weight_y_index = weight_z_index + weight_y * filter_length;
			//Uses y to calculate y index in value from array
			//Don't need to add padding cus we whant to get the index in the top left corner of kernal
			//Then add weight_y to get to the corect neuron for that weight
			const size_t from_y_start_index = from_z_start_index + ((y_id * stride + weight_y) * this_length);

			for (size_t weight_x = 0; weight_x < filter_length; ++weight_x)
			{
				value += weights[weight_y_index + weight_x] * values_in[from_y_start_index + weight_x + x_id * stride];
			}
		}
	}
	values_out[to_value_y_index + x_id] = value; 
}
//Think it's faster to split calc gradient and update weight cus can optimize memory acces beter
__kernel void UpdateWeightsConv(__global float* weights, __global float* delta_weights, __global float* values, __global float* next_gradient,
								  int stride, int length, int filter_length, float eta, float momentum)
{
	//x * length + y
	const size_t weight_xy = get_global_id(0);
	const size_t weight_z = get_global_id(1);
	const size_t filter_index = get_global_id(2);
	const size_t filter_length_pow = get_global_size(0);
	const size_t depth = get_global_size(1);
	const size_t num_filters = get_global_size(2);
    
	const size_t weight_x = weight_xy % filter_length;
	const size_t weight_y = (weight_xy - weight_x) / filter_length;
	const size_t next_length = (length - filter_length) / stride + 1;
	const size_t next_length_pow = next_length * next_length;
	const size_t length_pow = length * length;
	const size_t num_weights_filter = filter_length_pow * depth;
	
	//Calculating first to_value index for this filter
	const size_t to_value_z_index =  weight_z * next_length_pow;
	//Calculating index of first weight for this filter
	const unsigned weight_start_index = filter_index * num_weights_filter;
	//Calculating weight z index in array
	const unsigned weight_z_index = weight_start_index + weight_z * filter_length_pow;
	//Calculating weight z index in array
	const unsigned weight_y_index = weight_z_index + weight_y * filter_length;
	
	float new_delta_weight = 0.0f;
    
	for (unsigned y = 0; y < next_length; y++)
	{
		for (unsigned x = 0; x < next_length; x++)
		{
			//Calculating first to_value index for this filter
			const unsigned to_value_z_index = filter_index * next_length_pow;
			//Calculating to_value index in array using to_z_start_index
			const unsigned to_value_y_index = to_value_z_index + y * next_length;
			
			const unsigned from_z_start_index = weight_z * length_pow;
			const unsigned from_y_start_index = from_z_start_index + ((y  * stride + weight_y) * length);
    
			new_delta_weight +=
				eta
				* values[from_y_start_index + weight_x + x * stride]
				* next_gradient[to_value_y_index + x];
    
		}
	}
    
	new_delta_weight = new_delta_weight / next_length_pow
		+ momentum
		* delta_weights[weight_y_index + weight_x];
    
	delta_weights[weight_y_index + weight_x] = new_delta_weight;
	weights[weight_y_index + weight_x] += new_delta_weight;
}

__kernel void CalcGradientConvLinear(__global float* weights, __global float* values, __global float* this_gradients, __global float* next_gradients, 
									int stride, int filter_length, int num_filters)
{
	const size_t x_id = get_global_id(0);
	const size_t y_id = get_global_id(1);
	const size_t weight_z_id = get_global_id(2);
	const size_t this_length = get_global_size(0);
	const size_t this_depth = get_global_size(2);

	const size_t this_length_pow = this_length * this_length;
	const size_t next_length = (this_length - filter_length) / stride + 1;
	const size_t next_length_pow = next_length * next_length;
	const size_t next_length_no_stride = (this_length - filter_length) + 1;
	const size_t filter_length_pow = filter_length * filter_length;
	const size_t num_weights_filter = filter_length_pow * this_depth;
	
	const size_t this_z_start_index = weight_z_id * this_length_pow;
	const size_t this_y_start_index = this_z_start_index + y_id * this_length;
	const size_t this_gradient_index = this_y_start_index + x_id;

	this_gradients[this_gradient_index] = 0.0f;

	for (size_t filter_index = 0; filter_index < num_filters; ++filter_index)
	{
		//Calculating index of first weight for this filter
		const size_t weight_start_index = filter_index * num_weights_filter;
		//Calculating weight z index in array
		const size_t weight_z_index = weight_start_index + weight_z_id * filter_length_pow;
		//Calculating first to_value index for this filter
		const size_t next_value_z_index = filter_index * next_length_pow;

		for (size_t weight_y = 0; weight_y < filter_length; ++weight_y)
		{
			//position whitout stride. It needs to be devided with stride to get corect index in next layer
			const int next_y = y_id - weight_y;

			{//Continue if there is no neuron at that position in next layer
				if ((next_y % stride) != 0 || next_y < 0 || next_y >= next_length_no_stride)
					continue;
			}

			const size_t weight_y_index = weight_z_index + weight_y * filter_length;
			const size_t next_value_y_index = next_value_z_index + next_y / stride * next_length;

			for (unsigned weight_x = 0; weight_x < filter_length; ++weight_x)
			{
				//position whitout stride. It needs to be devided  with stride to get corect index in next layer
				const int next_x = x_id - weight_x;

				{//Continue if there is no neuron at that position in next layer
					if ((next_x % stride) != 0 || next_x < 0 || next_x >= next_length_no_stride)
						continue;
				}

				this_gradients[this_gradient_index] += weights[weight_y_index + weight_x] * next_gradients[next_value_y_index + next_x / stride];
			}
		}
	}
	this_gradients[this_gradient_index] = this_gradients[this_gradient_index] * values[this_gradient_index];
}
__kernel void CalcGradientConvRelu(__global float* weights, __global float* values, __global float* this_gradients, __global float* next_gradients, 
									int stride, int filter_length, int num_filters)
{
	const size_t x_id = get_global_id(0);
	const size_t y_id = get_global_id(1);
	const size_t weight_z_id = get_global_id(2);
	const size_t this_length = get_global_size(0);
	const size_t this_depth = get_global_size(2);

	const size_t this_length_pow = this_length * this_length;
	const size_t next_length = (this_length - filter_length) / stride + 1;
	const size_t next_length_pow = next_length * next_length;
	const size_t next_length_no_stride = (this_length - filter_length) + 1;
	const size_t filter_length_pow = filter_length * filter_length;
	const size_t num_weights_filter = filter_length_pow * this_depth;
	
	const size_t this_z_start_index = weight_z_id * this_length_pow;
	const size_t this_y_start_index = this_z_start_index + y_id * this_length;
	const size_t this_gradient_index = this_y_start_index + x_id;

	this_gradients[this_gradient_index] = 0.0f;

	for (size_t filter_index = 0; filter_index < num_filters; ++filter_index)
	{
		//Calculating index of first weight for this filter
		const size_t weight_start_index = filter_index * num_weights_filter;
		//Calculating weight z index in array
		const size_t weight_z_index = weight_start_index + weight_z_id * filter_length_pow;
		//Calculating first to_value index for this filter
		const size_t next_value_z_index = filter_index * next_length_pow;

		for (size_t weight_y = 0; weight_y < filter_length; ++weight_y)
		{
			//position whitout stride. It needs to be devided with stride to get corect index in next layer
			const int next_y = y_id - weight_y;

			{//Continue if there is no neuron at that position in next layer
				if ((next_y % stride) != 0 || next_y < 0 || next_y >= next_length_no_stride)
					continue;
			}

			const size_t weight_y_index = weight_z_index + weight_y * filter_length;
			const size_t next_value_y_index = next_value_z_index + next_y / stride * next_length;

			for (unsigned weight_x = 0; weight_x < filter_length; ++weight_x)
			{
				//position whitout stride. It needs to be devided  with stride to get corect index in next layer
				const int next_x = x_id - weight_x;

				{//Continue if there is no neuron at that position in next layer
					if ((next_x % stride) != 0 || next_x < 0 || next_x >= next_length_no_stride)
						continue;
				}

				this_gradients[this_gradient_index] += weights[weight_y_index + weight_x] * next_gradients[next_value_y_index + next_x / stride];
			}
		}
	}
	this_gradients[this_gradient_index] = this_gradients[this_gradient_index] * ReluDerivative(values[this_gradient_index]);
}