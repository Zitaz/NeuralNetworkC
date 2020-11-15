#include "../NeuralNetCode/NeuralNet.h"
#include "../NeuralNetCode/OpenCLFunctions.h"
#include "SDLFunctions.h"

#include <fstream>
#include <iostream>
#include <Windows.h>

//Example Code
//
//Using the mnist database
//http://yann.lecun.com/exdb/mnist/

void TrainNet();
Net::Types::TrainingDataVector LoadData(bool);
void RenderData();
Net::NeuralNet* InitNet();
unsigned ConvertToLittleEndian(unsigned char*);

void TrainNet()
{
	const unsigned num_passes = 1;

	Net::NeuralNet* net = InitNet();

	Net::Types::TrainingDataVector training_data = LoadData(false);

	Net::NetFunc::LoadNet(*net, "../Serialized/MnistNet");
	for (size_t i = 0; i < num_passes; i++)
	{
		//Net::Utility::ShuffleTrainingData(&training_data);
		Net::NetFunc::TrainNet(*net, training_data, 0.01f, 0.5f, 1000);
	}
	Net::NetFunc::SaveNet(*net, "../Serialized/MnistNet");

	delete net;
}

Net::Types::TrainingDataVector LoadData(bool load_test_data = false)
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

	Net::Types::TrainingDataVector training_data_vector;
	training_data_vector.reserve(num_data);

	for (unsigned i = 0; i < num_data; i++)
	{
		Net::Types::TrainingData training_data;

		training_data._output.resize(10);
		training_data._output[label_data[i + start_index_label]] = 1.0f;

		training_data._input.reserve(image_size);

		for (unsigned image_index = 0; image_index < image_size; image_index++)
		{
			training_data._input.push_back((unsigned char)(image_data[i * image_size + start_index_image + image_index]) / 255.0f);
		}

		training_data_vector.push_back(training_data);
	}

	return training_data_vector;
}

void RenderData()
{
	Net::NeuralNet* net = InitNet();

	Net::Types::TrainingDataVector training_data = LoadData(true);
	Net::NetFunc::LoadNet(*net, "../Serialized/MnistNet");

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
						data_index = training_data.size() - 1;
				}
				else
				{
					data_index++;

					if (data_index >= training_data.size())
						data_index = 0;
				}

				Sleep(250);

				window.DrawImage(training_data[data_index]._input, 0, 0, 28, 28, false);
				window.Render();

				std::cout << "Displaying image: " << data_index << std::endl;

				Net::NetFunc::FeedForward(*net, training_data[data_index]._input);

				Net::Types::LayerValues values = Net::NetFunc::GetOutputValues(*net);

				std::cout << "Output: " << std::endl;

				int highest_index = 0;

				for (unsigned i = 0; i < values.size(); i++)
				{
					if (values[highest_index] < values[i])
						highest_index = i;

					std::cout << values[i] << "  |  ";
				}

				std::cout << std::endl << "Best prediction: " << highest_index << std::endl;
			}
		}

		window.Update();

		Sleep(5);
	}

	window.DenitializeSDL();
	delete net;
}

Net::NeuralNet* InitNet()
{
	OpenCL::InitializeData();

	Net::NeuralNet* net = new Net::NeuralNet(5);

	Net::InitData::NetConvInitData layer_0_data = { Net::Types::ActivationFunction::LEAKY_RELU, 28, 1, 5, 32, 1 };
	Net::InitData::NetConvInitData layer_1_data = { Net::Types::ActivationFunction::LEAKY_RELU, 24, 32, 9, 96, 3 };
	Net::InitData::NetConvInitData layer_2_data = { Net::Types::ActivationFunction::LEAKY_RELU, 6, 96, 5, 96, 1 };
	Net::InitData::NetFCInitData layer_3_data = { Net::Types::ActivationFunction::LEAKY_RELU, 384, 10 };
	Net::InitData::NetOutputInitData layer_4_data = { Net::Types::ActivationFunction::LINEAR, 10 };

	Net::NetFunc::AddConvLayer(*net, layer_0_data, true);
	Net::NetFunc::AddConvLayer(*net, layer_1_data, true);
	Net::NetFunc::AddConvLayer(*net, layer_2_data, true);
	Net::NetFunc::AddFCLayer(*net, layer_3_data, true);
	Net::NetFunc::AddOutputLayer(*net, layer_4_data, true);

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