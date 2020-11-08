set game_dir=%~dp0
set flatbuffers_dir=%game_dir%\Flatbuffers
cd %flatbuffers_dir%

REM ~~~~~~~~~~~~~~~~ Generate binaries from premade (handmade) JSONs ~~~~~~~~~~~~~~~~
for %%a in (.\premade_*.json) do ".\flatc" --binary -o "..\Serialized" ".\Entity.fbs" "%%a"
call:CHECK_FAIL

REM ~~~~~~~~~~~~~~~~ Generate binaries from maya made JSONs ~~~~~~~~~~~~~~~~
for %%a in (.\level_*.json) do ".\flatc" --binary -o "..\Serialized" ".\LevelDef.fbs" "%%a"
call:CHECK_FAIL

REM ~~~~~~~~~~~~~~~~ Generate binaries from other JSONs ~~~~~~~~~~~~~~~~
".\flatc" --binary -o "..\..\game\data\serialized" ".\RenderingConstants.fbs" "RenderingConstants.json"
call:CHECK_FAIL
pause
exit

:: /// check if the app has failed
:CHECK_FAIL
@echo off
if NOT ["%errorlevel%"]==["0"] (
    echo.
    echo ~~~~~~~~~~~~~~ xXx_you_HaVe_aN_eRRoR_xXx ~~~~~~~~~~~~~~
    echo.
    pause
    exit /b %errorlevel%
)
@echo on