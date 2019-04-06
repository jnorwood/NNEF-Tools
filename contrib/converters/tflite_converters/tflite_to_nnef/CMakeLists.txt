
cmake_minimum_required(VERSION 3.13)
include(FetchContent)
include(ExternalProject)
set(CMAKE_BUILD_TYPE Debug)

option(ENABLE_TESTS "Enable tests" ON)
if (${ENABLE_TESTS})
    enable_testing()
endif()

FetchContent_Declare(
  flatbuffer
  URL https://github.com/google/flatbuffers/archive/v1.8.0.zip
  URL_HASH MD5=bb1557d1e8aabf2178c3c878ed8ff538
  SOURCE_DIR  flatbuffer
)
FetchContent_Declare(
  mobilenetv1
  URL http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz
  URL_HASH MD5=36af340c00e60291931cb30ce32d4e86
  SOURCE_DIR  mobilenetv1
)

FetchContent_Declare(
  mobilenetv2
  URL http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz
  URL_HASH MD5=04b4cc2536e1a51b9d29e98921b4970b
  SOURCE_DIR  mobilenetv2
)

FetchContent_Declare(
  tflite_schema
  URL https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema_generated.h
  SOURCE_DIR  tflite_schema
)

FetchContent_Declare(
  inceptionv1
  URL http://download.tensorflow.org/models/inception_v1_224_quant_20181026.tgz
  SOURCE_DIR  inceptionv1
)

FetchContent_Declare(
  inceptionv2
  URL http://download.tensorflow.org/models/inception_v2_224_quant_20181026.tgz
  SOURCE_DIR  inceptionv2
)

FetchContent_Declare(
  inceptionv3
  URL http://download.tensorflow.org/models/tflite_11_05_08/inception_v3_quant.tgz
  SOURCE_DIR  inceptionv3
)

FetchContent_Declare(
  inceptionv4
  URL http://download.tensorflow.org/models/inception_v4_299_quant_20181026.tgz
  SOURCE_DIR  inceptionv4
)


FetchContent_GetProperties(flatbuffer)
if(NOT flatbuffer_POPULATED)
  FetchContent_Populate(flatbuffer)
endif()

FetchContent_GetProperties(mobilenetv1)
if(NOT mobilenetv1_POPULATED)
  FetchContent_Populate(mobilenetv1)
endif()

FetchContent_GetProperties(mobilenetv2)
if(NOT mobilenetv2_POPULATED)
  FetchContent_Populate(mobilenetv2)
endif()

FetchContent_GetProperties(inceptionv1)
if(NOT inceptionv1_POPULATED)
  FetchContent_Populate(inceptionv1)
endif()

FetchContent_GetProperties(inceptionv2)
if(NOT inceptionv2_POPULATED)
  FetchContent_Populate(inceptionv2)
endif()

FetchContent_GetProperties(inceptionv3)
if(NOT inceptionv3_POPULATED)
  FetchContent_Populate(inceptionv3)
endif()

FetchContent_GetProperties(inceptionv4)
if(NOT inceptionv4_POPULATED)
  FetchContent_Populate(inceptionv4)
endif()

# doesn't work.  We have the old one anyway
#FetchContent_GetProperties(tflite_schema)
#if(NOT tflite_schema_POPULATED)
#  FetchContent_Populate(tflite_schema)
#endif()
add_definitions(-std=c++11)

# Source directories
set(DL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dl)
set(SCHEMA_DIR ${DL_DIR}/tflite)
set(FLATBUFFERS_DIR ${CMAKE_CURRENT_BINARY_DIR}/flatbuffer/include)
set(QUANTIZED_MODELS_DIR ${CMAKE_CURRENT_BINARY_DIR})

include_directories( ${FLATBUFFERS_DIR} ${SCHEMA_DIR})
file(GLOB LIB_HEADERS ${FLATBUFFERS_DIR}/*.h  ${SCHEMA_DIR}/*.h)
set(target tflite_to_nnef)
add_executable(${target} ${LIB_HEADERS} main.cpp)

add_test (NAME convert_mobilenetV1
	COMMAND ${CMAKE_CURRENT_BINARY_DIR}/tflite_to_nnef mobilenetv1/mobilenet_v1_1.0_224_quant.tflite mobv1_out
	)
add_test (NAME convert_mobilenetV2
	COMMAND ${CMAKE_CURRENT_BINARY_DIR}/tflite_to_nnef mobilenetv2/mobilenet_v2_1.0_224_quant.tflite mobv2_out
	)
add_test (NAME convert_inceptionv1
	COMMAND ${CMAKE_CURRENT_BINARY_DIR}/tflite_to_nnef inceptionv1/inception_v1_224_quant.tflite inceptionv1_out
	)
add_test (NAME convert_inceptionv2
	COMMAND ${CMAKE_CURRENT_BINARY_DIR}/tflite_to_nnef inceptionv2/inception_v2_224_quant.tflite inceptionv2_out
	)
add_test (NAME convert_inceptionv3
	COMMAND ${CMAKE_CURRENT_BINARY_DIR}/tflite_to_nnef inceptionv3/inception_v3_quant.tflite inceptionv3_out
	)
add_test (NAME convert_inceptionv4
	COMMAND ${CMAKE_CURRENT_BINARY_DIR}/tflite_to_nnef inceptionv4/inception_v4_299_quant.tflite inceptionv4_out
	)
