
cd "%~dp0flatbuffers"
for %%a in (.\*.fbs) do ".\flatc" --cpp -o "..\ModularNeuralNetworks\Code\Flatbuffers\Generated" "%%a"
cd "..\..\"
pause
exit