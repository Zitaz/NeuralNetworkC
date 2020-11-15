#pragma once

#include <vector>
#include <SDL.h>
#undef main

class Sprite;

//Example Code
class SDLUtilityClass
{
public:
	SDLUtilityClass();
	~SDLUtilityClass();

	void InitializeSDL(int width, int height, const char* title);
	void DenitializeSDL();

	void Update();
	void Render();

	void Render(Sprite sprite);

	void SetColor(float r, float g, float b, float a);

	void DrawRect(const SDL_Rect& rect);
	void FillRect(const SDL_Rect& rect);

	void DrawPoint(int x, int y);

	void DrawImage(float* pixel_data, int position_x, int position_y, int with, int height, bool color);

	float GetDeltaTime() { return _delta_time; }

	static void QuitSDL();

	SDL_Window* _sdl_window;
	SDL_Renderer* _sdl_renderer;
	SDL_Surface* _sdl_screen_surface;

	SDL_Point _mouse_location;

private:

	unsigned int _time;
	float _delta_time;

};

class Sprite
{
public:

	Sprite();
	Sprite(const char* Path, int x, int y, int h, int w);
	~Sprite();

	void Create(const char* Path, int x, int y, int h, int w);

	void LoadFromDisk(const char* Path);

	void SetRect(SDL_Rect rect) { _rect = rect; }
	void SetRect(int x, int y, int h, int w);

	void SetPos(int x, int y);
	void SetOriginPos(int x, int y);

	SDL_Surface* _image;
	SDL_Rect _rect;

private:

};