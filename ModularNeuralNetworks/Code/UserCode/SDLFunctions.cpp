
#include "SDLFunctions.h"
#include <stdio.h>


SDLUtilityClass::SDLUtilityClass()
{
	_time = 0;
	_delta_time = 0.0f;
}

SDLUtilityClass::~SDLUtilityClass()
{
}

void SDLUtilityClass::InitializeSDL(int width, int height, const char* title)
{
	//Initialize SDL
	if (SDL_Init(SDL_INIT_EVERYTHING) < 0)
	{
		printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
	}
	else
	{
		//Create window
		//data_instance._sdl_window = SDL_CreateWindow("Recording output window", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, x_output_size * display_pixel_x, y_output_size * display_pixel_y, SDL_WINDOW_SHOWN);
		SDL_CreateWindowAndRenderer(width, height, 0, &_sdl_window, &_sdl_renderer);
		SDL_SetRenderDrawBlendMode(_sdl_renderer, SDL_BLENDMODE_BLEND);
		SDL_SetWindowTitle(_sdl_window, title);

		if (_sdl_window == NULL)
		{
			printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
		}
		else
		{
			//Get window surface
			_sdl_screen_surface = SDL_GetWindowSurface(_sdl_window);

			//Fill the surface white
			SDL_FillRect(_sdl_screen_surface, NULL, SDL_MapRGB(_sdl_screen_surface->format, 0xFF, 0xFF, 0xFF));
		}

	}
}

void SDLUtilityClass::DenitializeSDL()
{
	//Destroy window
	SDL_DestroyWindow(_sdl_window);
}

void SDLUtilityClass::Update()
{
	//delta_time will be in secounds
	_delta_time = (SDL_GetTicks() - _time) / 1000.0f;

	_time = SDL_GetTicks();

	SDL_Event event;

	while (SDL_PollEvent(&event))
	{
		if (event.type == SDL_MOUSEMOTION) {
			_mouse_location.x = event.motion.x;
			_mouse_location.y = event.motion.y;
		}
	}
}

void SDLUtilityClass::Render() 
{
	//SDL_UpdateWindowSurface(_sdl_window);
	//SDL_FillRect(_sdl_screen_surface, NULL, SDL_MapRGB(_sdl_screen_surface->format, 0xFF, 0xFF, 0xFF));
	//SDL_BlitSurface(_sdl_screen_surface, nullptr, _sdl_screen_surface, nullptr);
	SDL_RenderPresent(_sdl_renderer);
}

void SDLUtilityClass::Render(Sprite sprite)
{
	SDL_BlitSurface(sprite._image, nullptr, _sdl_screen_surface, &sprite._rect);
}

void SDLUtilityClass::SetColor(float r, float g, float b, float a)
{
	SDL_SetRenderDrawColor(_sdl_renderer, r * 255, g * 255, b * 255, a * 255);
}

void SDLUtilityClass::DrawRect(const SDL_Rect& rect)
{
	SDL_RenderDrawRect(_sdl_renderer, &rect);
}

void SDLUtilityClass::FillRect(const SDL_Rect& rect)
{
	SDL_RenderFillRect(_sdl_renderer, &rect);
}

void SDLUtilityClass::DrawPoint(int x, int y)
{
	SDL_RenderDrawPoint(_sdl_renderer, x, y);
}

//Todo: Super slow, try and find a faster way
void SDLUtilityClass::DrawImage(float* pixel_data, int position_x, int position_y, int width, int height, bool color)
{
	const unsigned image_size = width * height;

	for (size_t y = 0; y < height; y++)
	{
		const unsigned y_index = y * width;

		for (size_t x = 0; x < width; x++)
		{
			const unsigned x_index = y_index + x;

			float red;
			float green;
			float blue;

			if (color) {
				red = pixel_data[x_index] * 255.0f;
				green = pixel_data[image_size + x_index] * 255.0f;
				blue = pixel_data[image_size * 2 + x_index] * 255.0f;
			}
			else {
				red = green = blue = pixel_data[x_index] * 255.0f;
			}
			
			SDL_SetRenderDrawColor(_sdl_renderer, red, green, blue, 255.0f);

			//Write to screen
			SDL_RenderDrawPoint(_sdl_renderer, x + position_x, y + position_y);
		}
	}
}

void SDLUtilityClass::QuitSDL()
{
	//Quit SDL subsystems
	SDL_Quit();
}



Sprite::Sprite()
{
	_image = nullptr;
}

Sprite::Sprite(const char* Path, int x, int y, int h, int w) 
{
	_rect.x = x;
	_rect.y = y;
	_rect.h = h;
	_rect.w = w;

	_image = SDL_LoadBMP(Path);

	if (_image == nullptr)
		printf(SDL_GetError());
}

Sprite::~Sprite()
{

	//delete m_pImage;
	//delete m_pRect;

}

void Sprite::Create(const char * Path, int x, int y, int h, int w)
{
	_rect.x = x;
	_rect.y = y;
	_rect.h = h;
	_rect.w = w;

	_image = SDL_LoadBMP(Path);

	if (_image == nullptr)
		printf(SDL_GetError());
}

void Sprite::LoadFromDisk(const char* Path) 
{
	_image = SDL_LoadBMP(Path);

	if (_image == nullptr)
		printf(SDL_GetError());

}
void Sprite::SetRect(int x, int y, int h, int w) 
{
	_rect.x = x;
	_rect.y = y;
	_rect.h = h;
	_rect.w = w;

}

void Sprite::SetPos(int x, int y) {

	_rect.x = x;
	_rect.y = y;

}
void Sprite::SetOriginPos(int x, int y) {

	_rect.x = x - _rect.w / 2;
	_rect.y = y - _rect.h / 2;

}
