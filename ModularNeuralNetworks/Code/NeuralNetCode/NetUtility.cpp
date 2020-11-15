#include "NetUtility.h"

namespace Net {

	namespace Utility {

		//void ShuffleTrainingData(Types::TrainingDataVector* training_data)
		//{
		//	std::shuffle(std::begin(*training_data), std::end(*training_data), random_device);
		//}

		float GetDerivativ(Net_ActivationFunction function, float value)
		{
			switch (function)
			{
			case Net_ActivationFunction::NET_ACTIVATION_FUNC_LEAKY_RELU:
				return ReluDerivative(value);
			case Net_ActivationFunction::NET_ACTIVATION_FUNC_SIGMOID:
				return SigmoidDerivative(value);
			case Net_ActivationFunction::NET_ACTIVATION_FUNC_LINEAR:
				return 1.0f;
			default:
				return 0.0f;
			}
		}

		//std::vector<float> RescaleImage(std::vector<float> image_data, int width, int height, int new_width, int new_height, bool has_color)
		//{
		//	std::vector<float> new_image_data(new_width * new_height * (has_color * 3));
		//
		//	const float scale_x = (float)new_width / (float)width;
		//	const float scale_y = (float)new_height / (float)height;
		//
		//	const int num_pixels = width * height;
		//	const int num_pixels_new = new_width * new_height;
		//
		//	int num_x_to_scale_x = 0;
		//	int num_y_to_scale_y = 0;
		//
		//	int x_scaled_prev = -1;
		//	int y_scaled_prev = -1;
		//
		//	for (size_t y = 0; y < height; y++)
		//	{
		//		int y_scaled = y * scale_y;
		//		int change_num_prev_pixels_y = 0;
		//
		//		if (y_scaled - y_scaled_prev > 1) {
		//			change_num_prev_pixels_y = y_scaled - y_scaled_prev - 1;
		//		}
		//
		//		if (y_scaled != y_scaled_prev) {//calc num_y_to_scale_y
		//			int added_y = 0;
		//
		//			y_scaled_prev = y_scaled;
		//
		//			while (true)
		//			{
		//				added_y++;
		//
		//				int test_y = (added_y + y) * scale_y;
		//
		//				if (test_y != y_scaled_prev || (added_y + y) >= height) {
		//					num_y_to_scale_y = added_y;
		//					break;
		//				}
		//			}
		//		}
		//
		//		for (size_t x = 0; x < width; x++)
		//		{
		//			int x_scaled = x * scale_x;
		//			int change_num_prev_pixels_x = 0;
		//
		//			if (x_scaled - x_scaled_prev > 1) {
		//				change_num_prev_pixels_x = x_scaled - x_scaled_prev - 1;
		//			}
		//
		//			if (x_scaled != x_scaled_prev) {//calc num_x_to_scale_x
		//				int added_x = 0;
		//
		//				x_scaled_prev = x_scaled;
		//
		//				while (true)
		//				{
		//					added_x++;
		//
		//					int test_x = (added_x + x) * scale_x;
		//
		//					if (test_x != x_scaled_prev || (added_x + x) >= width) {
		//						num_x_to_scale_x = added_x;
		//						break;
		//					}
		//				}
		//			}
		//
		//			for (unsigned y_prev = 0; y_prev < change_num_prev_pixels_y + 1; y_prev++)
		//			{
		//				for (unsigned x_prev = 0; x_prev < change_num_prev_pixels_x + 1; x_prev++)
		//				{
		//					for (unsigned color = 0; color < 3; color++)
		//					{
		//						new_image_data[color * num_pixels_new + (y_scaled - y_prev) * new_width + (x_scaled - x_prev)] += image_data[color * num_pixels + y * width + x] / (num_x_to_scale_x * num_y_to_scale_y);
		//
		//						if (!has_color)
		//							break;
		//					}
		//				}
		//			}
		//
		//		}
		//	}
		//
		//	return new_image_data;
		//}
	}
}
