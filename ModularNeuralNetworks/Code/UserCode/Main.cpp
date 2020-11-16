extern "C" {
#include "../NeuralNetCode/NeuralNet.h"
}

#include "SDLFunctions.h"

#include <fstream>
#include <iostream>
#include <Windows.h>

//Example Code
//
//Using the mnist database
//http://yann.lecun.com/exdb/mnist/

void TrainNet();
Net_ArrayTrainingData LoadData(bool);
void RenderData();
Net_NeuralNet* InitNet();
unsigned ConvertToLittleEndian(unsigned char*);

void TrainNet()
{
	const unsigned num_passes = 1;

	Net_NeuralNet* net = InitNet();

	Net_ArrayTrainingData training_data = LoadData(false);

	Net_LoadNet(net, "../Serialized/MnistNet");

	for (size_t i = 0; i < num_passes; i++)
	{
		//ShuffleTrainingData(&training_data);
		Net_TrainNet(net, &training_data, 0.01f, 0.5f, 1000);
	}
	Net_SaveNet(net, "../Serialized/MnistNet");
	
	Net_FreeArrayTD(&training_data);

	Net_FreeLayersNet(net);
	delete net;
}

Net_ArrayTrainingData LoadData(bool load_test_data = false)
{
	const int start_index_image = 16;
	const int start_index_label = 8;

	char image_path[100];
	char lable_path[100];

	if (load_test_data)
	{
		strcpy_s(image_path, "../serialized/mnist/t10k-images.idx3-ubyte");
		strcpy_s(lable_path, "../serialized/mnist/t10k-labels.idx1-ubyte");
	}
	else
	{
		strcpy_s(image_path, "../serialized/mnist/train-images.idx3-ubyte");
		strcpy_s(lable_path, "../serialized/mnist/train-labels.idx1-ubyte");
	}

	std::ifstream image_stream(image_path, std::ifstream::binary);
	std::string image_data(std::istreambuf_iterator<char>(image_stream), (std::istreambuf_iterator<char>()));

	std::ifstream label_stream(lable_path, std::ifstream::binary);
	std::string label_data(std::istreambuf_iterator<char>(label_stream), (std::istreambuf_iterator<char>()));

	unsigned num_data = ConvertToLittleEndian((unsigned char*)&image_data[4]);
	unsigned image_size = ConvertToLittleEndian((unsigned char*)&image_data[12]) * ConvertToLittleEndian((unsigned char*)&image_data[8]);

	Net_ArrayTrainingData training_data_vector;
	Net_CreateArrayTD(&training_data_vector, num_data, 0);

	for (unsigned i = 0; i < num_data; i++)
	{
		Net_TrainingData training_data;
		Net_CreateArrayF(&training_data._output, 10, 10);
		Net_CreateArrayF(&training_data._input, image_size, 0);
		Net_ZeroArrayF(&training_data._output);

		training_data._output._data[label_data[i + start_index_label]] = 1.0f;

		for (unsigned image_index = 0; image_index < image_size; image_index++)
		{
			Net_AddArrayF(&training_data._input, (unsigned char)(image_data[i * image_size + start_index_image + image_index]) / 255.0f);
		}

		Net_AddArrayTD(&training_data_vector, training_data);
	}

	return training_data_vector;
}

void RenderData()
{
	Net_NeuralNet* net = InitNet();

	Net_ArrayTrainingData training_data = LoadData(true);
	Net_LoadNet(net, "../Serialized/MnistNet");

	SDLUtilityClass window;
	window.InitializeSDL(28, 28, "TM");

	int data_index = 0;

	std::cout << "Press \"S\" to quit" << std::endl;

	while (GetKeyState('S') >= 0)
	{
		{//Change image
			if (GetKeyState(VK_RIGHT) < 0 || GetKeyState(VK_LEFT) < 0)
			{
				if (GetKeyState(VK_LEFT) < 0)
				{
					data_index--;

					if (data_index < 0)
						data_index = training_data._size - 1;
				}
				else
				{
					data_index++;

					if (data_index >= training_data._size)
						data_index = 0;
				}

				Sleep(250);

				window.DrawImage(training_data._data[data_index]._input._data, 0, 0, 28, 28, false);
				window.Render();

				std::cout << "Displaying image: " << data_index << std::endl;

				Net_FeedForward(net, training_data._data[data_index]._input);

				Net_ArrayF values = Net_GetOutputValues(net);

				std::cout << "Output: " << std::endl;

				int highest_index = 0;

				for (unsigned i = 0; i < values._size; i++)
				{
					if (values._data[highest_index] < values._data[i])
						highest_index = i;

					std::cout << values._data[i] << "  |  ";
				}

				Net_FreeArrayF(&values);

				std::cout << std::endl << "Best prediction: " << highest_index << std::endl;
			}
		}

		window.Update();

		Sleep(5);
	}

	Net_FreeArrayTD(&training_data);

	window.DenitializeSDL();

	Net_FreeLayersNet(net);
	delete net;
}

Net_NeuralNet* InitNet()
{
	Net_NeuralNet* net = new Net_NeuralNet;

	Net_CreateNet(net, 5);

	Net_ConvInitData layer_0_data = { Net_ActivationFunction::NET_ACTIVATION_FUNC_LEAKY_RELU, 28, 1, 5, 32, 1 };
	Net_ConvInitData layer_1_data = { Net_ActivationFunction::NET_ACTIVATION_FUNC_LEAKY_RELU, 24, 32, 9, 96, 3 };
	Net_ConvInitData layer_2_data = { Net_ActivationFunction::NET_ACTIVATION_FUNC_LEAKY_RELU, 6, 96, 5, 96, 1 };
	Net_FCInitData layer_3_data = { Net_ActivationFunction::NET_ACTIVATION_FUNC_LEAKY_RELU, 384, 10 };
	Net_OutputInitData layer_4_data = { Net_ActivationFunction::NET_ACTIVATION_FUNC_LINEAR, 10 };

	Net_AddConvLayer(net, &layer_0_data, true);
	Net_AddConvLayer(net, &layer_1_data, true);
	Net_AddConvLayer(net, &layer_2_data, true);
	Net_AddFCLayer(net, &layer_3_data, true);
	Net_AddOutputLayer(net, &layer_4_data, true);

	return net;
}

unsigned ConvertToLittleEndian(unsigned char * bytes)
{
	return (unsigned)((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]);
}

int main()
{
	std::cout << "Press \"F1\" to train" << std::endl;
	std::cout << "Press \"F2\" to display" << std::endl;


	while (true)
	{
		if (GetAsyncKeyState(VK_F1)) 
		{
			TrainNet();
			break;
		}
		else if(GetAsyncKeyState(VK_F2))
		{
			RenderData();
			break;
		}

		Sleep(5);
	}

	return 0;
}