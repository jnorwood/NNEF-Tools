# tflite_to_nnef converter

 
## Dependencies
This uses google flatbuffers v1.80, which is currently compatible with this development.
There is a recent update of the tensorflow lite schema_generated.h that has moved ahead to flatbuffers 1.9.
It may not be compatible with this current build, so the v1.8 compatible schema_generated.h is being included in this branch. 
Some of the code is duplicated from armnn's tflite converter, and so its license is provided
The only working conversions are for the quantized versions of MobilenetV1 and MobilenetV2, Inception v1, v2, v3, 
and these will be installed by the cmake script.

On ubuntu 18.04, the use of filesystem I'b building with gcc 7.3  
 
CMAKE is required in MSVC to generate the .sln file

The installation script for the dependencies is dowload_dependencies.sh, which you should run in a bash shell
It is tested in the gitbash shell on Windows.
## Installation
Clone this repository to your computer and execute the following shell commands:
```sh
cd NNEF-Tools
git checkout tflite_to_nnef
cd  contrib/converters/tflite_converters/tflite_to_nnef
 
mkdir build
cd build
cmake ..
make
ctest
```
If using MSVC, create the .sln file using cmake,  I've stopped  testing with MSVC for now
```sh
in MSVC17 menus select Tools->Visual Studio Command Prompt
cd \NNEF-Tools\contrib\converters\tflite_converters\tflite_to_nnef
mkdir build
cd build
cmake -G"Visual Studio 15 2017 Win64" ..

```
 
## Usage
the CMakeLists.txt has examples that can execute with ctest

the download progress can be turned off by commenting out the set(FETCHCONTENT_QUIET OFF) line in CMakeLists.txt

tflite_to_nnef quantized_filename.tflite output_path
creates a graph.nnef, graph.quant and subdirectories with the weight and bias .dat quantized binary files
```sh
jay@jay-sk:~/nnefx_tst/NNEF-Tools/contrib/converters/tflite_converters/tflite_to_nnef/build$ cmake ..
-- The C compiler identification is GNU 7.3.0
-- The CXX compiler identification is GNU 7.3.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: /home/jay/nnefx_tst/NNEF-Tools/contrib/converters/tflite_converters/tflite_to_nnef/build
jay@jay-sk:~/nnefx_tst/NNEF-Tools/contrib/converters/tflite_converters/tflite_to_nnef/build$ make
Scanning dependencies of target tflite_to_nnef
[ 50%] Building CXX object CMakeFiles/tflite_to_nnef.dir/main.cpp.o
[100%] Linking CXX executable tflite_to_nnef
[100%] Built target tflite_to_nnef
jay@jay-sk:~/nnefx_tst/NNEF-Tools/contrib/converters/tflite_converters/tflite_to_nnef/build$ ctest
Test project /home/jay/nnefx_tst/NNEF-Tools/contrib/converters/tflite_converters/tflite_to_nnef/build
    Start 1: convert_mobilenetV1
1/6 Test #1: convert_mobilenetV1 ..............   Passed    0.33 sec
    Start 2: convert_mobilenetV2
2/6 Test #2: convert_mobilenetV2 ..............   Passed    0.28 sec
    Start 3: convert_inceptionv1
3/6 Test #3: convert_inceptionv1 ..............   Passed    0.50 sec
    Start 4: convert_inceptionv2
4/6 Test #4: convert_inceptionv2 ..............   Passed    0.85 sec
    Start 5: convert_inceptionv3
5/6 Test #5: convert_inceptionv3 ..............   Passed    1.79 sec
    Start 6: convert_inceptionv4
6/6 Test #6: convert_inceptionv4 ..............   Passed    3.18 sec

100% tests passed, 0 tests failed out of 6

Total Test time (real) =   6.93 sec
```
 