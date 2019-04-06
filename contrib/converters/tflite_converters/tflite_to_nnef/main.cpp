#include <iostream> // C++ header file for printing
#include <fstream> // C++ header file for file access
#include <string>
#ifdef WIN32
#include <filesystem>
namespace fs = std::experimental::filesystem;
#else
// will be used for mkdir call
#include <sys/stat.h>
#endif
#include "flatbuffers/util.h"
#include "schema_generated.h"

#pragma warning(disable:4996)
#ifdef WIN32
#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#endif
#ifndef BOOLEAN
#define BOOLEAN int
#endif
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

enum class DataType
{
    Float16 = 0,
    Float32 = 1,
    QuantisedAsymm8 = 2,
    Signed32 = 3
};


// testable helpers


using ModelPtr = std::unique_ptr<tflite::ModelT>;
using SubGraphPtr = std::unique_ptr<tflite::SubGraphT>;
using OperatorPtr = std::unique_ptr<tflite::OperatorT>;
using OperatorCodePtr = std::unique_ptr<tflite::OperatorCodeT>;
using TensorPtr = std::unique_ptr<tflite::TensorT>;
using TensorRawPtr = const tflite::TensorT *;
using TensorRawPtrVector = std::vector<TensorRawPtr>;
using TensorIdRawPtr = std::pair<size_t, TensorRawPtr>;
using TensorIdRawPtrVector = std::vector<TensorIdRawPtr>;
using BufferPtr = std::unique_ptr<tflite::BufferT>;
using BufferRawPtr = const tflite::BufferT *;

ModelPtr                              m_Model;
std::vector<std::string> uint8WeightFilenames;
std::vector<std::string> int32BiasFilenames;
std::vector<std::string> parameterConstantFilenames;
std::string lastOutName = "";

// signature for the parser functions
using OperatorParsingFunction = void( *)(size_t subgraphIndex, size_t operatorIndex);


static ModelPtr LoadModelFromFile(const char * fileName);
static ModelPtr LoadModelFromBinary(const uint8_t * binaryContent, size_t len);
static TensorRawPtrVector GetInputs(const ModelPtr & model, size_t subgraphIndex, size_t operatorIndex);
static TensorRawPtrVector GetOutputs(const ModelPtr & model, size_t subgraphIndex, size_t operatorIndex);
static TensorIdRawPtrVector GetSubgraphInputs(const ModelPtr & model, size_t subgraphIndex);
static TensorIdRawPtrVector GetSubgraphOutputs(const ModelPtr & model, size_t subgraphIndex);
static std::vector<int32_t>& GetInputTensorIds(const ModelPtr& model, size_t subgraphIndex, size_t operatorIndex);
static std::vector<int32_t>& GetOutputTensorIds(const ModelPtr& model, size_t subgraphIndex, size_t operatorIndex);
//
static BufferRawPtr GetBuffer(const ModelPtr& model, size_t bufferIndex);
static size_t maxFilterSz = 0;
//static TensorInfo OutputShapeOfSqueeze(const std::vector<uint32_t> & squeezeDims,
//    const TensorInfo & inputTensorInfo);

void ParseUnsupportedOperator(size_t subgraphIndex, size_t operatorIndex);
void ParseAveragePool2D(size_t subgraphIndex, size_t operatorIndex);
void ParseConv2D(size_t subgraphIndex, size_t operatorIndex);
void ParseAdd(size_t subgraphIndex, size_t operatorIndex);
void ParseDepthwiseConv2D(size_t subgraphIndex, size_t operatorIndex);
void ParseSoftmax(size_t subgraphIndex, size_t operatorIndex);
void ParseSqueeze(size_t subgraphIndex, size_t operatorIndex);
void ParseReshape(size_t subgraphIndex, size_t operatorIndex);
void ParseMaxPool(size_t subgraphIndex, size_t operatorIndex);
void ParseConcat(size_t subgraphIndex, size_t operatorIndex);
void ParseMean(size_t subgraphIndex, size_t operatorIndex);
void ParseFullyConnected(size_t subgraphIndex, size_t operatorIndex);


// These will be one per layer, although we would need an input2Names if doing concat
int opIndex = 0;
std::vector<std::string> pathNames;
std::vector<std::pair<int, std::string>> tensorRename;
std::vector<std::pair<int, int>> filterDw;
std::vector<std::pair<int, std::string>> filterPathRename;
std::vector<std::pair<int, std::string>> filterFolderRename;
std::vector<std::pair<int, std::string>> biasPathRename;
std::vector<std::pair<int, std::string>> biasFolderRename;
std::stringstream externDefs;
std::stringstream filterDefs;
std::stringstream biasDefs;
std::stringstream netDefs;
std::stringstream graphHeader;
std::stringstream quantDefs; // for the quantized data output

// This section supports rename of files from the tflite names to the NNEF convention
bool isRenamed(int inputBufIndex) {
    for (auto p : tensorRename) {
        if (p.first == inputBufIndex) {
            return true;
        }
    }
    return false;
}
bool isFilterDw(int inputBufIndex) {
    for (auto p : filterDw) {
        if (p.first == inputBufIndex) {
            return true;
        }
    }
    return false;
}
std::string getRename(int inputBufIndex) {
    for (auto p : tensorRename) {
        if (p.first == inputBufIndex) {
            return p.second;
        }
    }
    return "";
}

std::string getFilterPathRename(int inputBufIndex) {
    for (auto p : filterPathRename) {
        if (p.first == inputBufIndex) {
            return p.second;
        }
    }
    return "";
}
std::string getFilterFolderRename(int inputBufIndex) {
    for (auto p : filterFolderRename) {
        if (p.first == inputBufIndex) {
            return p.second;
        }
    }
    return "";
}

std::string getBiasPathRename(int inputBufIndex) {
    for (auto p : biasPathRename) {
        if (p.first == inputBufIndex) {
            return p.second;
        }
    }
    return "";
}
std::string getBiasFolderRename(int inputBufIndex) {
    for (auto p : biasFolderRename) {
        if (p.first == inputBufIndex) {
            return p.second;
        }
    }
    return "";
}

// tensorRename.push_back(std::pair<std::string, int>(intval, stringval));


void CalcPadding(uint32_t inputSize,
    uint32_t filterSize,
    uint32_t stride,
    uint32_t& paddingFront,
    uint32_t& paddingBack,
    tflite::Padding padding)
{
    paddingFront = 0;
    paddingBack = 0;

    //auto out_height = ceil(float(inputSize) / float(stride));

    //auto out_width = ceil(float(inputSize) / float(stride));

    if (padding == tflite::Padding_SAME)
    {
        uint32_t outputSize = (inputSize + stride - 1) / stride;
        uint32_t temp = (outputSize - 1) * stride + filterSize;
        if (temp > inputSize)
        {
            paddingFront = (temp - inputSize) / 2;
            paddingBack = (temp - inputSize) - paddingFront;
        }
    }
}
std::vector<unsigned int> AsUnsignedVector(const std::vector<int32_t> & in)
{
    std::vector<unsigned int> result;
    result.reserve(in.size());
    for (auto & i : in)
    {
        result.push_back(i);
    }
    return result;
}

// TODO this should output some error.  It arrives here for any currently unsupported operation
// in this conversion utility
void ParseUnsupportedOperator(size_t subgraphIndex, size_t operatorIndex) {
    return;
};

// This is saving to a format compliant with an NNEF  quant file
void outputFilterQuantizationInfo(std::stringstream &s, TensorRawPtr tp, std::string &nm, int bits, float downscale) {
    auto q = tp->quantization.get();
    float max = 0;
    float min = 0;
    float scale = 0;
    int64_t zp = 0;
    size_t max_sz = q->max.size();
    size_t min_sz = q->min.size();
    size_t scale_sz = q->scale.size();
    size_t zp_sz = q->zero_point.size();
    auto type = tp->type; // TensorType_INT32 for bias, 
    if (max_sz > 0)
        max = q->max[0];
    if (min_sz > 0)
        min = q->min[0];
    if (scale_sz > 0)
        scale = q->scale[0];
    if (zp_sz > 0)
        zp = q->zero_point[0];
    auto shp = tp->shape;
    size_t  shp_sz = shp.size();
    int32_t shpn[5] = { 0,0,0,0,0 };
    for (size_t j = 0; j < shp_sz && j < 5; j++) {
        shpn[j] = shp[j];
    }
    s << std::setprecision(12) << std::fixed << " # quant: min:" << min << " <= scale:" << scale << " * (q - zoff:" << zp << ") <= max:" << max << "\n";
    if (bits == 8)
    quantDefs << std::setprecision(12) <<std::fixed << "\"" << nm << "\": tflite_quantize(min = "  << min << ", max = "  << max 
        << " ,scale =" << scale
        << " ,downscale =" << downscale
        << " ,offset =" << zp << " ,bits = " << bits << ");\n";

}

// TODO, we are currently converting avgpool to mean_reduce for MobilenetV1 and Squeezenet
// but we may need avgpool elsewhere. Need to have an option for this conversion.
void ParseAveragePool2D(size_t subgraphIndex, size_t operatorIndex) {
    // we're going to convert this to mean_reduce if out_ch==in_ch and out_h=out_w=1, since that is what is done by nnef

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsPool2DOptions();


    auto stride_w = options->stride_w; // we need to output these if not [1,1]
    auto stride_h = options->stride_h; // output as second to last param as stride = [2,2], for example
    auto filter_h = options->filter_height;
    auto filter_w = options->filter_width;
    auto padding = options->padding;


    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    auto tp0 = inputs[0]; // input tensor

    // assuming input is NHWC
    //unsigned int inputHeight = inputTensorInfo.GetShape()[1];
    //unsigned int inputWidth = inputTensorInfo.GetShape()[2];
    auto inputHeight = tp0->shape[1];  
    auto inputWidth = tp0->shape[2];
    auto inputChannel = tp0->shape[3];
    uint32_t padTop, padBottom, padLeft, padRight;
    padTop = padBottom = padLeft = padRight = 0;


    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    auto tp3 = outputs[0];
    auto outputHeight = tp3->shape[1];  // for first conv_dw this is [1,112,112,32]
                                        //unsigned int inputWidth = inputTensorInfo.GetShape()[2];
    auto outputWidth = tp3->shape[2];
    auto outputChannel = tp3->shape[3];

    std::stringstream s;
    BOOLEAN is_mean_reduce = FALSE;
    CalcPadding(inputHeight, filter_h, stride_h, padTop, padBottom, options->padding);
    CalcPadding(inputWidth, filter_w, stride_w, padLeft, padRight, options->padding);

    if (padLeft == 0 && padRight==0 && padTop==0 && padBottom==0 && outputChannel == inputChannel && outputWidth == 1 && outputHeight == 1) {
        // need to also check padding 0
        // if so then this is going to be substituted with mean_reduce?
        // this is for the end that nnef substitutes
        s << "mean_reduce" << subgraphIndex << "_" << operatorIndex << "";
        is_mean_reduce = TRUE;

    }
    else {
        s << "avg_pool" << subgraphIndex << "_" << operatorIndex << "";

    }
    std::string nm = s.str();


    auto inputBufIndex = tp0->buffer;
    std::string inName;
    if (isRenamed(inputBufIndex)) {
        inName = getRename(inputBufIndex);
    }
    else {
        inName = tp0->name;// shouldn't get here?
        }
    std::string outName;
    auto outBufIndex = tp3->buffer;
    if (isRenamed(outBufIndex)) {
        outName = getRename(outBufIndex);
    }
    else {
        s.str("");
        if (is_mean_reduce) {
            s << "reduce" << opIndex;
        }
        else {
            s << "avgpool" << opIndex;
        }
        outName = s.str();
        tensorRename.push_back(std::pair<int, std::string>(outBufIndex, outName));
    }

    if (is_mean_reduce) {
        netDefs << outName << " = mean_reduce(input = " << inName << ", axes = [2,3]);";

        netDefs << " # in = [1," << inputChannel << " ," << inputHeight << " ," << inputWidth <<
            " ], out= [1, " << outputChannel << " ," << outputHeight << " ," << outputWidth <<
            " ], ";
    }
    else {
        // process avgpool just like maxpool
        netDefs << outName << " = avg_pool(input = " << inName << ", size = [1, 1, " <<
            filter_h << ", " << filter_w << "]";
        netDefs << ", padding = [(0, 0), (0, 0), (" <<
            padTop << ", " <<
            padBottom << "), (" <<
            padLeft << ", " <<
            padRight << ")]";
        if (stride_h != 1) {
            netDefs << ", stride = [1, 1, " << stride_w << ", " << stride_h << "]";
        }
        netDefs << ");";
        netDefs << " # in = [1," << inputChannel << " ," << inputHeight << " ," << inputWidth <<
            " ], out= [1, " << outputChannel << " ," << outputHeight << " ," << outputWidth <<
            " ]";

    }
    outputFilterQuantizationInfo(netDefs, tp3, outName,8, 0.0f);
    lastOutName=outName;
    opIndex++;

};

// This is NNEF compatible output of quantization info for bias, which uses int32 instead
// of uint8 used by the weights.  NNEF supports multiple quantization fragment defintions
// There will be a custom handler in the custom parser for these quantization fragments
// so that we can handle them at code generation time rather than at runtime. 
void outputBiasQuantizationInfo(std::stringstream &s, TensorRawPtr tp, std::string &nm, int bits) {
    auto q = tp->quantization.get();
    float max = 0;
    float min = 0;
    float scale = 0;
    int64_t zp = 0;
    size_t max_sz = q->max.size();
    size_t min_sz = q->min.size();
    size_t scale_sz = q->scale.size();
    size_t zp_sz = q->zero_point.size();
    auto type = tp->type; // TensorType_INT32 for bias, 
    if (max_sz > 0)
        max = q->max[0];
    if (min_sz > 0)
        min = q->min[0];
    if (scale_sz > 0)
        scale = q->scale[0];
    if (zp_sz > 0)
        zp = q->zero_point[0];
    auto shp = tp->shape;
    size_t  shp_sz = shp.size();
    int32_t shpn[5] = { 0,0,0,0,0 };
    for (size_t j = 0; j < shp_sz && j < 5; j++) {
        shpn[j] = shp[j];
    }
    s << std::setprecision(12) << std::fixed << " # quant: scale:" << scale << " * q\n";

    // TODO, what to do for scale, offset, multipliers? can this be some other fragment?
    if (bits == 32)
    quantDefs << std::setprecision(12) << std::fixed << "\"" << nm << "\": tflite_bias_quantize(scale = " << scale << " ,bits = " << bits << ");\n";

}
// gets here via the operations jump table
void ParseAdd(size_t subgraphIndex, size_t operatorIndex) {
    float inputScale = 1.0f;
    float inputScaleB = 1.0f;
    float outputScale = 1.0f;
    float outputDownScale = 1.0f;
    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsAddOptions();
    if (options != NULL) {
        // is fused used?
        auto x = options->fused_activation_function;
        if (x !=  0) {

        }
    }
    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    auto in_sz = inputs.size(); // should be 2

    // outputs has is_variable = false
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    auto out_sz = outputs.size();

    auto tp0 = inputs[0]; // input A
    auto tp1 = inputs[1]; // input B
    auto inputAHeight = tp0->shape[1];  // for first conv_dw this is [1,224,224,3]
    auto inputAWidth = tp0->shape[2];
    auto inputAChannel = tp0->shape[3];
    auto inputBHeight = tp1->shape[1];  // for first conv_dw this is [1,224,224,3]
    auto inputBWidth = tp1->shape[2];
    auto inputBChannel = tp1->shape[3];
    std::string outName;

    auto tp3 = outputs[0];
    auto outBufIndex = tp3->buffer;
    std::stringstream s;
    s << "add" << subgraphIndex << "_" << operatorIndex;
    std::string nm = s.str();
    if (isRenamed(outBufIndex)) {
        outName = getRename(outBufIndex);
    }
    else {
        s.str("");
        s << "add" << opIndex;
        outName = s.str();
        tensorRename.push_back(std::pair<int, std::string>(outBufIndex, outName));
    }

    auto inputBufIndex = tp0->buffer;
    std::string inNameA;
    if (isRenamed(inputBufIndex)) {
        inNameA = getRename(inputBufIndex);
    }
    else {
        inNameA = tp0->name;
        if (inNameA == "input") {
            s.str("");
            if (opIndex > 0)
                s << "data" << opIndex;
            else s << "data";
            inNameA = s.str();
            tensorRename.push_back(std::pair<int, std::string>(inputBufIndex, inNameA));
            //[1,224,224,3] -> [1, 3, 224, 224]
            externDefs << inNameA << " = external(shape = [ 1, " <<
                tp0->shape[3] << ", " <<
                tp0->shape[1] << ", " <<
                tp0->shape[2] << "]);";
            std::string dat("data");
            //TODO check this tp0 ... probably need the data output vector
            auto q = tp0->quantization.get();
            inputScale = q->scale[0];

            outputFilterQuantizationInfo(externDefs, tp0, dat, 8, 0.0f);

        }
    }
    auto inputBufIndexB = tp1->buffer;
    std::string inNameB;
    if (isRenamed(inputBufIndexB)) {
        inNameB = getRename(inputBufIndexB);
    }
    else {
        inNameB = tp1->name;
        if (inNameB == "input") {
            s.str("");
            if (opIndex > 0)
                s << "data" << opIndex;
            else s << "data";
            inNameB = s.str();
            tensorRename.push_back(std::pair<int, std::string>(inputBufIndexB, inNameB));
            //[1,224,224,3] -> [1, 3, 224, 224]
            externDefs << inNameB << " = external(shape = [ 1, " <<
                tp1->shape[3] << ", " <<
                tp1->shape[1] << ", " <<
                tp1->shape[2] << "]);";
            std::string dat("data");
            //TODO check this tp0 ... probably need the data output vector
            auto q = tp1->quantization.get();
            inputScaleB = q->scale[0];

            outputFilterQuantizationInfo(externDefs, tp1, dat, 8, 0.0f);

        }
    }

    netDefs << outName << " = add(x = " << inNameA << ", y = " << inNameB << ");";

    auto outputHeight = tp3->shape[1];
                                        //unsigned int inputWidth = inputTensorInfo.GetShape()[2];
    auto outputWidth = tp3->shape[2];
    auto outputChannel = tp3->shape[3];
    netDefs << " # x = [1," << inputAChannel << " ," << inputAHeight << " ," << inputAWidth <<
        " ], y = [1," << inputBChannel << ", " << inputBHeight << ", " << inputBWidth <<
        " ], z = [1, " << outputChannel << " ," << outputHeight << " ," << outputWidth <<
        " ]";
    // TODO here needs to output also the downscale value
    // So probably need the third quantize definition? or else a non zero downscale param
    auto q = tp3->quantization.get();
    outputScale = q->scale[0];
    inputScale = tp0->quantization->scale[0];
    // B is from the residual bypass
    inputScaleB = tp1->quantization->scale[0];
    outputDownScale = inputScale / outputScale;
    outputFilterQuantizationInfo(netDefs, tp3, outName, 8, outputDownScale);

    // Here, let's tack on the input and output shapes for now.
    lastOutName = outName;
    opIndex++;
}
// gets here via the operations jump table
void ParseConv2D(size_t subgraphIndex, size_t operatorIndex) {
    float inputScale = 1.0f;
    float outputScale = 1.0f;
    float kernelScale = 1.0f;
    float outputDownScale = 1.0f;
    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsConv2DOptions();


    auto stride_w = options->stride_w; // we need to output these if not [1,1]
    auto stride_h = options->stride_h; // output as second to last param as stride = [2,2], for example
    // options also has a fused activation type flag
    // options also has dilation_h and dilation_w parameters

    // inputs has input, kernel weights and bias, where kernel_weights are uint8 and bias are int32
    // The bias dimensions will need to be changed for output to nnef, to prepend with batch 1
    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    auto in_sz = inputs.size();

    // outputs has is_variable = false
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    auto out_sz = outputs.size();

    auto tp0 = inputs[0]; // input tensor
    auto tp1 = inputs[1]; // weights tensor
    if (in_sz == 3) {
        auto tp2 = inputs[2]; // bias tensor
    }

    // assuming input is NHWC
    //unsigned int inputHeight = inputTensorInfo.GetShape()[1];
    //unsigned int inputWidth = inputTensorInfo.GetShape()[2];
    auto inputHeight = tp0->shape[1];  // for first conv_dw this is [1,224,224,3]
    auto inputWidth = tp0->shape[2];
    auto inputChannel = tp0->shape[3];


    // assuming the filter is OHWI : Output, H, W, Input, but tf is HWIO and tflite? OHWI for regular conv
    // but for NNEF filter, we need OIHW, so conversion would be [0,3,1,2] for conv weights.
    // for Depthwise sep (first) conv weights, order is 1HWI, but we need for NNEF OIHW, so [3,0,1,2]??
    // for Depthwise sep (second) conv weights, order is OHWI, but we need for NNEF OIHW, so [3,0,1,2]??
    // which is essentially the same as NHWC = IHWO
    //unsigned int filterHeight = filterTensorInfo.GetShape()[1];
    //unsigned int filterWidth = filterTensorInfo.GetShape()[2];
    // [32, 3, 3, 3] on first conv
    auto filterOutCh = tp1->shape[0]; //
    auto filterHeight = tp1->shape[1]; //
    auto filterWidth = tp1->shape[2];
    auto filterInChannel = tp1->shape[3];

    uint32_t padTop, padBottom, padLeft, padRight;
    CalcPadding(inputHeight, filterHeight, stride_h, padTop, padBottom, options->padding);
    CalcPadding(inputWidth, filterWidth, stride_w, padLeft, padRight, options->padding);
    BOOLEAN is_sep = FALSE;
    if (filterWidth == 1 && filterHeight == 1 && operatorIndex > 0) {
        // check to see if second half of depthwise separable
        const auto & operatorPtr_prev = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex-1];
        const auto * options_prev = operatorPtr_prev->builtin_options.AsConv2DOptions();
        auto const & opCodePtr = m_Model->operator_codes[operatorPtr_prev->opcode_index];
        auto builtinCode_prev = opCodePtr->builtin_code;
        if (builtinCode_prev == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) {
            is_sep = TRUE;
        }

    }

    std::string sep = (is_sep) ? "_sep" : "";
    std::stringstream s;
    s << "conv" << subgraphIndex << "_" << operatorIndex << sep;
    std::string nm = s.str();


    if (inputs.size() == 3)
    {
        auto tp2 = inputs[2];
    }
    else
    {
     }


    auto inputBufIndex = tp0->buffer;
    std::string inName;
    if (isRenamed(inputBufIndex)) {
        inName = getRename(inputBufIndex);
    }
    else {
        inName = tp0->name;
        if (inName == "input") {
            s.str("");
            if (opIndex > 0)
                s << "data" << opIndex;
            else s << "data";
            inName = s.str();
            tensorRename.push_back(std::pair<int,std::string>(inputBufIndex, inName));
            //[1,224,224,3] -> [1, 3, 224, 224]
            externDefs << inName << " = external(shape = [ 1, " <<
                tp0->shape[3] << ", " <<
                tp0->shape[1] << ", " <<
                tp0->shape[2] << "]);";
            std::string dat("data");
            //TODO check this tp0 ... probably need the data output vector
            auto q = tp0->quantization.get();
            inputScale = q->scale[0];
            
            outputFilterQuantizationInfo(externDefs, tp0, dat, 8, 0.0f);

        }
    }

    std::string filterName;
    std::string filterPath;
    std::string filterFolder;
    auto filterBufIndex = tp1->buffer;
    if (isRenamed(filterBufIndex)) {
        filterName = getRename(filterBufIndex);
        filterPath = getFilterPathRename(filterBufIndex);
        filterFolder = getFilterFolderRename(filterBufIndex);
    }
    else {
        s.str("");
        s << "filter" << opIndex;
        filterName = s.str();
        tensorRename.push_back(std::pair<int, std::string>(filterBufIndex, filterName));
        s.str("");
        s << "conv" << opIndex << sep;
        filterFolder = s.str();
        filterFolderRename.push_back(std::pair<int, std::string>(filterBufIndex, filterFolder));
        s << "/filter";
        filterPath = s.str();
        filterPathRename.push_back(std::pair<int, std::string>(filterBufIndex, filterPath));

        //[1,224,224,3] -> [1, 3, 224, 224], so [och, h,w, ich] -> [och,ich,h,w] for filters
        // but in case of depthwise
        filterDefs << filterName << " = variable(shape = [" <<
            tp1->shape[0] << ", " <<
            tp1->shape[3] << ", " <<
            tp1->shape[1] << ", " <<
            tp1->shape[2] << "], label = '" << filterPath << "');";
        auto q = tp1->quantization.get();
        kernelScale = q->scale[0];

        outputFilterQuantizationInfo(filterDefs,tp1, filterName,8,0.0f);
        size_t filterSz = tp1->shape[0] * tp1->shape[1] * tp1->shape[2] * tp1->shape[3];
        if (filterSz > maxFilterSz) 
            maxFilterSz = filterSz;


    }

    std::string biasName = "";
    std::string biasPathName;
    std::string biasFolderName;
    if (inputs.size() == 3)
    {
        auto tp2 = inputs[2];
        auto biasBufIndex = tp2->buffer;
        if (isRenamed(biasBufIndex)) {
            biasName = getRename(biasBufIndex);
            biasPathName = getBiasPathRename(biasBufIndex);
            biasFolderName = getBiasFolderRename(biasBufIndex);
        }
        else {
            s.str("");
            s << "bias" << opIndex;
            biasName = s.str();
            tensorRename.push_back(std::pair<int, std::string>(biasBufIndex, biasName));
            s.str("");
            s << "conv" << opIndex << sep;
            biasFolderName = s.str();
            biasFolderRename.push_back(std::pair<int, std::string>(biasBufIndex, biasFolderName));
            s << "/bias";
            biasPathName = s.str();
            biasPathRename.push_back(std::pair<int, std::string>(biasBufIndex, biasPathName));
            //[1,224,224,3] -> [1, 3, 224, 224]
            biasDefs << biasName << " = variable(shape = [1, " <<
                tp2->shape[0] << "], label = '" << biasPathName << "');";
            outputBiasQuantizationInfo(biasDefs, tp2, biasName, 32);
        }
    }
    std::string outName;
    auto tp3 = outputs[0];
    auto outBufIndex = tp3->buffer;
    if (isRenamed(outBufIndex)) {
        outName = getRename(outBufIndex);
    }
    else {
        s.str("");
        s << "conv" << opIndex;
        outName = s.str();
        tensorRename.push_back(std::pair<int, std::string>(outBufIndex, outName));
    }

    s.str("");
    std::string opName = "conv";
    s << "conv" << opIndex;
    s.str("");
    s << "conv" << opIndex << sep << "/filter";
    filterPath = s.str();
    // we will create a std::pair with this index to rename the buffer

    netDefs << outName << " = conv(input = " << inName << ", filter = " << filterName <<
        ", bias = " << biasName << ", padding = [(" <<
        padTop << ", " <<
        padBottom << "), (" <<
        padLeft << ", " <<
        padRight << ")]";
    if (stride_h != 1) {
        netDefs << ", stride = [" << stride_w << ", " << stride_h << "]";
    }
    netDefs << ");";

    auto outputHeight = tp3->shape[1];  // for first conv_dw this is [1,112,112,32]
                                        //unsigned int inputWidth = inputTensorInfo.GetShape()[2];
    auto outputWidth = tp3->shape[2];
    auto outputChannel = tp3->shape[3];

    netDefs << " # in = [1," << inputChannel << " ," << inputHeight << " ," << inputWidth <<
        " ], out= [1, " << outputChannel << " ," << outputHeight << " ," << outputWidth <<
        " ]";
    // TODO here needs to output also the downscale value
    // So probably need the third quantize definition? or else a non zero downscale param
    // TODO why are some of these scale sizes 0?  could be a schema difference.
    auto q = tp3->quantization.get();
    outputScale = (q->scale.size()==0)?1.0 : q->scale[0];
    inputScale = (tp0->quantization->scale.size()==0)? 1.0: tp0->quantization->scale[0];
    outputDownScale = inputScale * kernelScale / outputScale;
    outputFilterQuantizationInfo(netDefs, tp3, outName,8, outputDownScale);

    // Here, let's tack on the input and output shapes for now.
    lastOutName = outName;
    opIndex++;
    return;
};

void ParseDepthwiseConv2D(size_t subgraphIndex, size_t operatorIndex) {

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsDepthwiseConv2DOptions();
    float inputScale = 1.0f;
    float outputScale = 1.0f;
    float kernelScale = 1.0f;
    float outputDownScale = 1.0f;

    auto stride_w = options->stride_w; // we need to output these if not [1,1]
    auto stride_h = options->stride_h; // output as second to last param as stride = [2,2], for example
    auto depth_multiplier = options->depth_multiplier;

    // inputs[0] is input from prior layer
    // inputs[1] is uint8 weights.  there is a buffer number here, ref to a buffer
    // inputs[2] is the folded bias, int32, but I think we need it
    // we can get to the buffers through the subgraphptr in the call to GetInputs
    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    auto in_sz = inputs.size();
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    auto out_sz = outputs.size();

//    TensorInfo inputTensorInfo = ToTensorInfo(inputs[0]);
//    TensorInfo filterTensorInfo = ToTensorInfo(inputs[1]);

    auto tp0 = inputs[0]; // input tens
    auto tp1 = inputs[1]; // weight tens
    if (in_sz == 3) {
        auto tp2 = inputs[2]; // bias tens
    }


    // assuming input is NHWC
 
    //auto inputHeight = inputTensorInfo.GetShape()[1];
    // TensorFlow Lite uses BHWC, same as kNHWC TensorFlow activations

    auto inputHeight =tp0->shape[1];  // for first conv_dw this is [1,112,112,32]
    //unsigned int inputWidth = inputTensorInfo.GetShape()[2];
    auto inputWidth = tp0->shape[2];
    auto inputChannel = tp0->shape[3];
    // assuming the filter is OHWI : Output, H, W, Input, we need OIHW
    //unsigned int filterHeight = filterTensorInfo.GetShape()[1];
    // tensorflow lite k1HWO, // Our standard for DepthwiseConv weights, which is OHWI, since I=O, but 1 is O here
    // due to depthwise per channel convolution... so really only doing one output ch = one input ch per conv
    auto filterHeight = tp1->shape[1]; // for first dw this is [1,3,3,32], where 32 is out channel . we need [32,1,3,3] in filter
    auto filterWidth = tp1->shape[2];
    auto filterOutCh = tp1->shape[3]; // we don't need shape[0], which is 1 fo depth-wise?
    //unsigned int filterWidth = filterTensorInfo.GetShape()[2];

    uint32_t padTop, padBottom, padLeft, padRight;
    CalcPadding(inputHeight, filterHeight, stride_h, padTop, padBottom, options->padding);
    CalcPadding(inputWidth, filterWidth, stride_w, padLeft, padRight, options->padding);

    std::stringstream s;
    // in NNEF terms, this needs to count conv, not conv_dw?
    s << "conv" << subgraphIndex << "_" << operatorIndex << "_dw";
    std::string nm = s.str();
    // for NNEF, the layer needs to be the same for the DW and associated

    // nnef will require , groups = 0 parameter of regular conv for depth-wise indicator
    // The nnef to c++ generator will also need to use _dw on the filter names
    // The code generated will also need to handle the per channel execution of the convolution
    if (inputs.size() == 3)
    {
        auto tp2 = inputs[2];
    }
    else
    {
    }


    auto inputBufIndex = tp0->buffer;
    std::string inName;

    inputScale = tp0->quantization->scale[0];
    if (isRenamed(inputBufIndex)) {
        inName = getRename(inputBufIndex);
    }
    else {
        inName = tp0->name;
        if (inName == "input") {
            s.str("");
            if (opIndex > 0)
                s << "data" << opIndex;
            else s << "data";
            inName = s.str();
            tensorRename.push_back(std::pair<int, std::string>(inputBufIndex, inName));
            //[1,224,224,3] -> [1, 3, 224, 224]
            externDefs << inName << " = external(shape = [ 1, " <<
                tp0->shape[3] << ", " <<
                tp0->shape[1] << ", " <<
                tp0->shape[2] << "]);";
            std::string dat("data");
            //TODO check this tp0 ... probably need the data output vector
            auto q = tp0->quantization.get();
            inputScale = q->scale[0];

            outputFilterQuantizationInfo(externDefs, tp0, dat, 8, 0.0f);

        }
    }

    std::string filterName;
    std::string filterPathName;
    std::string filterFolderName;
    auto filterBufIndex = tp1->buffer;
    filterDw.push_back(std::pair<int, int>(filterBufIndex, 1));
    if (isRenamed(filterBufIndex)) {
        filterName = getRename(filterBufIndex);
        filterPathName = getFilterPathRename(filterBufIndex);
        filterFolderName = getFilterFolderRename(filterBufIndex);
    }
    else {
        s.str("");
        s << "filter" << opIndex;
        filterName = s.str();
        tensorRename.push_back(std::pair<int, std::string>(filterBufIndex, filterName));
        s.str("");
        s << "conv" << opIndex << "_dw";
        filterFolderName = s.str();
        filterFolderRename.push_back(std::pair<int, std::string>(filterBufIndex, filterFolderName));
        s << "/filter";
        filterPathName = s.str();
        filterPathRename.push_back(std::pair<int, std::string>(filterBufIndex, filterPathName));
        //[1,224,224,3] -> [1, 3, 224, 224]
        // This should be filterDw order so ->[3,`1,
        // for depthwise convert 1HWO to O1HW, for first dw [1,3,3,32]->[32,1,3,3]
        filterDefs << filterName << " = variable(shape = [" <<
            tp1->shape[3] << ", " <<
            tp1->shape[0] << ", " <<
            tp1->shape[1] << ", " <<
            tp1->shape[2] << "], label = '" << filterPathName << "');";
        kernelScale = tp1->quantization->scale[0];

        outputFilterQuantizationInfo(filterDefs, tp1, filterName,8,0.0f);
        size_t filterSz = tp1->shape[0] * tp1->shape[1] * tp1->shape[2] * tp1->shape[3];
        if (filterSz > maxFilterSz)
            maxFilterSz = filterSz;

    }

    std::string biasName;
    std::string biasPathName;
    std::string biasFolderName;
    if (inputs.size() == 3)
    {
        auto tp2 = inputs[2];
        auto biasBufIndex = tp2->buffer;
        if (isRenamed(biasBufIndex)) {
            biasName = getRename(biasBufIndex);
            biasPathName = getBiasPathRename(biasBufIndex);
            biasFolderName = getBiasFolderRename(biasBufIndex);
        }
        else {
            s.str("");
            s << "bias" << opIndex;
            biasName = s.str();
            tensorRename.push_back(std::pair<int, std::string>(biasBufIndex, biasName));
            s.str("");
            s << "conv" << opIndex << "_dw";
            biasFolderName = s.str();
            biasFolderRename.push_back(std::pair<int, std::string>(biasBufIndex, biasFolderName));
            s << "/bias";
            biasPathName = s.str();
            biasPathRename.push_back(std::pair<int, std::string>(biasBufIndex, biasPathName));
            //[1,224,224,3] -> [1, 3, 224, 224]
            biasDefs << biasName << " = variable(shape = [1, " <<
                tp2->shape[0] << "], label = '" << biasPathName << "');";
            outputBiasQuantizationInfo(biasDefs, tp2, biasName, 32);

        }
    }

    std::string outName;
    auto tp3 = outputs[0];
    auto outBufIndex = tp3->buffer;
    if (isRenamed(outBufIndex)) {
        outName = getRename(outBufIndex);
    }
    else {
        s.str("");
        s << "conv" << opIndex;
        outName = s.str();
        tensorRename.push_back(std::pair<int, std::string>(outBufIndex, outName));
    }

    netDefs << outName << " = conv(input = " << inName << ", filter = " << filterName <<
        ", bias = " << biasName << ", padding = [(" <<
        padTop << ", " <<
        padBottom << "), (" <<
        padLeft << ", " <<
        padRight << ")]";
    if (stride_h != 1) {
        netDefs << ", stride = [" << stride_w << ", " << stride_h << "]";
    }
    netDefs << ", groups = 0);";
    auto outputHeight = tp3->shape[1];  // for first conv_dw this is [1,112,112,32]
                                       //unsigned int inputWidth = inputTensorInfo.GetShape()[2];
    auto outputWidth = tp3->shape[2];
    auto outputChannel = tp3->shape[3];

    netDefs << " # in = [1," << inputChannel << " ," << inputHeight << " ," << inputWidth <<
        " ], out= [1, " << outputChannel << " ," << outputHeight << " ," << outputWidth <<
        " ], ";
    outputScale = tp3->quantization->scale[0];
    outputDownScale = inputScale * kernelScale / outputScale;
    outputFilterQuantizationInfo(netDefs, tp3, outName,8, outputDownScale);
    size_t filterSz = tp1->shape[0] * tp1->shape[1] * tp1->shape[2] * tp1->shape[3];
    if (filterSz > maxFilterSz)
        maxFilterSz = filterSz;


    s.str("");
    std::string opName = "conv";
    s << "conv" << opIndex;
    s.str("");
    s << "conv" << opIndex << "_dw" << "/filter";
    std::string filterPath = s.str();
    // we will create a std::pair with this index to rename the buffer
    lastOutName = outName;
    opIndex++;

};
void ParseSoftmax(size_t subgraphIndex, size_t operatorIndex) {
    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsSoftmaxOptions();

    auto m_beta = options->beta;

    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    auto in_sz = inputs.size();
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    auto out_sz = outputs.size();
    auto tp0 = inputs[0]; // input tensor
    auto inputChannel = tp0->shape[1];  // this should be 1001 [1,1001], same as output
    auto tp1 = outputs[0]; // output tensor
    auto outputChannel = tp1->shape[1]; // note this output is still a uint8, so maybe we did a conversion then back
    

    std::stringstream s;
    s << "softmax" << subgraphIndex << "_" << operatorIndex << "";
    std::string nm = s.str();

 
    auto inputBufIndex = tp0->buffer;
    std::string inName;
    if (isRenamed(inputBufIndex)) {
        inName = getRename(inputBufIndex);
    }
    else {
        inName = tp0->name;// shouldn't get here?
    }
    std::string outName;
    auto tp3 = outputs[0];
    auto outBufIndex = tp3->buffer;
    if (isRenamed(outBufIndex)) {
        outName = getRename(outBufIndex);
    }
    else {
        s.str("");
        s << "softmax" << opIndex;
        outName = s.str();
        tensorRename.push_back(std::pair<int, std::string>(outBufIndex, outName));
    }
    netDefs << outName << " = softmax(x = " << inName << ", axes = [1]);";


    netDefs << " # in = [1," << inputChannel << " ], out = [1, " << inputChannel << " ], ";
    outputFilterQuantizationInfo(netDefs, tp3, outName,8,0.0f);

    lastOutName = outName;
    opIndex++;

};
void ParseMean(size_t subgraphIndex, size_t operatorIndex) {
    // we're going to convert this to mean_reduce, since that is what is done by nnef

    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsReducerOptions();

    //auto xx = options->keep_dims;
    
   // auto stride_w = options->stride_w; // we need to output these if not [1,1]
   // auto stride_h = options->stride_h; // output as second to last param as stride = [2,2], for example
    //auto filter_h = options->filter_height;
   // auto filter_w = options->filter_width;
   // auto padding = options->padding;


    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    auto tp0 = inputs[0]; // input tensor

                          // assuming input is NHWC
                          //unsigned int inputHeight = inputTensorInfo.GetShape()[1];
                          //unsigned int inputWidth = inputTensorInfo.GetShape()[2];
    auto inputHeight = tp0->shape[1];
    auto inputWidth = tp0->shape[2];
    auto inputChannel = tp0->shape[3];
    uint32_t padTop, padBottom, padLeft, padRight;
    padTop = padBottom = padLeft = padRight = 0;


    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);

    std::stringstream s;
    s << "mean_reduce" << subgraphIndex << "_" << operatorIndex << "";
    std::string nm = s.str();


    auto inputBufIndex = tp0->buffer;
    std::string inName;
    if (isRenamed(inputBufIndex)) {
        inName = getRename(inputBufIndex);
    }
    else {
        inName = tp0->name;// shouldn't get here?
    }
    std::string outName;
    auto tp3 = outputs[0];
    auto outBufIndex = tp3->buffer;
    if (isRenamed(outBufIndex)) {
        outName = getRename(outBufIndex);
    }
    else {
        s.str("");
        s << "reduce" << opIndex;
        outName = s.str();
        tensorRename.push_back(std::pair<int, std::string>(outBufIndex, outName));
    }

    netDefs << outName << " = mean_reduce(input = " << inName << ", axes = [2,3]);";
    auto outputChannel = tp3->shape[1];  // for first conv_dw this is [1,112,112,32]
                                        //unsigned int inputWidth = inputTensorInfo.GetShape()[2];
    auto outputWidth = 1;
    auto outputHeight = 1;

    netDefs << " # in = [1," << inputChannel << " ," << inputHeight << " ," << inputWidth <<
        " ], out= [1, " << outputChannel << " ," << outputHeight << " ," << outputWidth <<
        " ], ";
    outputFilterQuantizationInfo(netDefs, tp3, outName, 8, 0.0f);
    lastOutName = outName;
    opIndex++;
}

void ParseSqueeze(size_t subgraphIndex, size_t operatorIndex) {
    // dont have anything for this,  I guess our Softmax is just dumb.
    return;
};
void ParseReshape(size_t subgraphIndex, size_t operatorIndex) {
    // dont have anything for this,  I guess our Reshape isn't needed is just dumb.
    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsSoftmaxOptions();


    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    auto in_sz = inputs.size();
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    auto out_sz = outputs.size();
    auto tp0 = inputs[0]; // input tensor
    auto inputHeight = tp0->shape[1];  // 
    auto inputWidth = tp0->shape[2];  // 
    auto inputChannel = tp0->shape[3];
    auto tp3 = outputs[0]; // output tensor
    int outputHeight = 0; // note this output is still a uint8, so maybe we did a conversion then back
    int outputWidth = 0;
    int outputChannel = 0;
    int outRank = tp3->shape.size();
    if (outRank == 2) {
        outputHeight = 1;
        outputWidth = 1;
        outputChannel = tp3->shape[1];
    }
    else if (outRank == 4) {
        outputHeight = tp3->shape[1]; // note this output is still a uint8, so maybe we did a conversion then back
        outputWidth = tp3->shape[2];
        outputChannel = tp3->shape[3];
    }


    std::stringstream s;
    s << "reshape" << subgraphIndex << "_" << operatorIndex << "";
    std::string nm = s.str();

 
    auto inputBufIndex = tp0->buffer;
    std::string inName;
    if (isRenamed(inputBufIndex)) {
        inName = getRename(inputBufIndex);
    }
    else {
        inName = tp0->name;// shouldn't get here?
    }
    std::string outName;
    auto outBufIndex = tp3->buffer;
    if (isRenamed(outBufIndex)) {
        outName = getRename(outBufIndex);
    }
    else {
        s.str("");
        s << "reshape" << opIndex;
        outName = s.str();
        tensorRename.push_back(std::pair<int, std::string>(outBufIndex, outName));
    }
    netDefs << outName << " = reshape(input = " << inName <<
        ", shape = [1," << outputChannel;
    if (outRank == 4)
        netDefs << "," << outputHeight << "," << outputWidth;
    netDefs << "]);";
                                         //unsigned int inputWidth = inputTensorInfo.GetShape()[2];

    netDefs << " # in = [1," << inputChannel << " ," << inputHeight << " ," << inputWidth <<
        " ], out= [1, " << outputChannel;
    if (outRank == 4)
        netDefs << " ," << outputHeight << " ," << outputWidth;
    
    netDefs <<
        " ], ";
    outputFilterQuantizationInfo(netDefs, tp3, outName, 8, 0.0f);
    //TODO should we do these?
    lastOutName = outName;
    opIndex++;
    return;
};
void ParseMaxPool(size_t subgraphIndex, size_t operatorIndex) {
    // dont have anything for this,  I guess our Reshape isn't needed is just dumb.
    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsPool2DOptions();

    auto stride_w = options->stride_w; // we need to output these if not [1,1]
    auto stride_h = options->stride_h; // output as second to last param as stride = [2,2], for example


    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    auto in_sz = inputs.size();


    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    auto out_sz = outputs.size();

    auto tp0 = inputs[0]; // input tensor
    auto inputHeight = tp0->shape[1];  // 
    auto inputWidth = tp0->shape[2];  // 
    auto inputChannel = tp0->shape[3];
    auto tp1 = outputs[0]; // output tensor
    auto outputHeight = tp1->shape[1]; // note this output is still a uint8, so maybe we did a conversion then back
    auto outputWidth = tp1->shape[2];
    auto outputChannel = tp1->shape[3];

    auto filterHeight = options->filter_height;
    auto filterWidth = options->filter_width;

    uint32_t padTop, padBottom, padLeft, padRight;
    CalcPadding(inputHeight, filterHeight, stride_h, padTop, padBottom, options->padding);
    CalcPadding(inputWidth, filterWidth, stride_w, padLeft, padRight, options->padding);

    std::stringstream s;
    s << "maxpool" << subgraphIndex << "_" << operatorIndex << "";
    std::string nm = s.str();


    auto inputBufIndex = tp0->buffer;
    std::string inName;
    if (isRenamed(inputBufIndex)) {
        inName = getRename(inputBufIndex);
    }
    else {
        inName = tp0->name;// shouldn't get here?
    }
    std::string outName;
    auto outBufIndex = tp1->buffer;
    if (isRenamed(outBufIndex)) {
        outName = getRename(outBufIndex);
    }
    else {
        outName = inName;
        s.str("");
        s << "maxpool" << opIndex;
        outName = s.str();
        tensorRename.push_back(std::pair<int, std::string>(outBufIndex, outName));
    }
    netDefs << outName << " = max_pool(input = " << inName << ", size = [1, 1, " <<
        filterHeight << ", " << filterWidth << "]";
    netDefs  << ", padding = [(0, 0), (0, 0), (" <<
        padTop << ", " <<
        padBottom << "), (" <<
        padLeft << ", " <<
        padRight << ")]";
    if (stride_h != 1) {
        netDefs << ", stride = [1, 1, " << stride_w << ", " << stride_h << "]";
    }
    netDefs << ");";
    netDefs << " # in = [1," << inputChannel << " ," << inputHeight << " ," << inputWidth <<
        " ], out= [1, " << outputChannel << " ," << outputHeight << " ," << outputWidth <<
        " ]";
    outputFilterQuantizationInfo(netDefs, tp1, outName, 8, 0.0f);

    //TODO not currently handling border or dilation
    lastOutName = outName;
    opIndex++;
    return;
};
void ParseConcat(size_t subgraphIndex, size_t operatorIndex) {
    // dont have anything for this,  I guess our Reshape isn't needed is just dumb.
    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsConcatenationOptions();

    auto axis = options->axis;
    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    auto in_sz = inputs.size();
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    auto out_sz = outputs.size();
    auto tp1 = outputs[0]; // output tensor
    auto outputHeight = tp1->shape[1]; // note this output is still a uint8, so maybe we did a conversion then back
                                        //unsigned int inputWidth = inputTensorInfo.GetShape()[2];
    auto outputWidth = tp1->shape[2];
    auto outputChannel = tp1->shape[3];


    std::stringstream s;
    s << "concat" << subgraphIndex << "_" << operatorIndex << "";
    std::string nm = s.str();


    auto tp0 = inputs[0]; // input tensor
    auto inputHeight = tp0->shape[1];  // 
    auto inputBufIndex = tp0->buffer;
    std::string inName;
    if (isRenamed(inputBufIndex)) {
        inName = getRename(inputBufIndex);
    }
    else {
        inName = tp0->name;// shouldn't get here?
    }


    std::string outName;
    auto outBufIndex = tp1->buffer;
    if (isRenamed(outBufIndex)) {
        outName = getRename(outBufIndex);
    }
    else {
        outName = inName;
        s.str("");
        s << "concat" << opIndex;
        outName = s.str();
        tensorRename.push_back(std::pair<int, std::string>(outBufIndex, outName));
    }


    netDefs << outName << " = concat([" << inName ;

    for (int j = 1; j < in_sz; j++) {
        auto tp0 = inputs[j]; // input tensor
        auto inputHeight = tp0->shape[1];  // 
        auto inputBufIndex = tp0->buffer;
        std::string inName;
        if (isRenamed(inputBufIndex)) {
            inName = getRename(inputBufIndex);
        }
        else {
            inName = tp0->name;// shouldn't get here?
        }
        netDefs << "," << inName ;
    }

    netDefs << "], axis = " << ((axis==3)? 1 : axis) << ");";

    netDefs << " #  out= [1, " << outputChannel << " ," << outputHeight << " ," << outputWidth <<
        " ]";
    outputFilterQuantizationInfo(netDefs, tp1, outName, 8, 0.0f);

    //TODO not currently handling border or dilation
    lastOutName = outName;
    opIndex++;
    return;
};



// gets here via the operations jump table
void ParseFullyConnected(size_t subgraphIndex, size_t operatorIndex) {
    float inputScale = 1.0f;
    float outputScale = 1.0f;
    float kernelScale = 1.0f;
    float outputDownScale = 1.0f;
    const auto & operatorPtr = m_Model->subgraphs[subgraphIndex]->operators[operatorIndex];
    const auto * options = operatorPtr->builtin_options.AsFullyConnectedOptions();


    auto stride_w = 1; // options->stride_w; // we need to output these if not [1,1]
    auto stride_h = 1; // options->stride_h; // output as second to last param as stride = [2,2], for example
                                       // options also has a fused activation type flag
                                       // options also has dilation_h and dilation_w parameters

                                       // inputs has input, kernel weights and bias, where kernel_weights are uint8 and bias are int32
                                       // The bias dimensions will need to be changed for output to nnef, to prepend with batch 1
    auto inputs = GetInputs(m_Model, subgraphIndex, operatorIndex);
    auto in_sz = inputs.size();

    // outputs has is_variable = false
    auto outputs = GetOutputs(m_Model, subgraphIndex, operatorIndex);
    auto out_sz = outputs.size();

    auto tp0 = inputs[0]; // input tensor
    auto tp1 = inputs[1]; // weights tensor
    if (in_sz == 3) {
        auto tp2 = inputs[2]; // bias tensor
    }

    // assuming input is NHWC
    //unsigned int inputHeight = inputTensorInfo.GetShape()[1];
    //unsigned int inputWidth = inputTensorInfo.GetShape()[2];
    auto inputHeight = tp0->shape[1];  // for first conv_dw this is [1,224,224,3]
    auto inputWidth = tp0->shape[2];
    auto inputChannel = tp0->shape[3];


    // assuming the filter is OHWI : Output, H, W, Input, but tf is HWIO and tflite? OHWI for regular conv
    // but for NNEF filter, we need OIHW, so conversion would be [0,3,1,2] for conv weights.
    // for Depthwise sep (first) conv weights, order is 1HWI, but we need for NNEF OIHW, so [3,0,1,2]??
    // for Depthwise sep (second) conv weights, order is OHWI, but we need for NNEF OIHW, so [3,0,1,2]??
    // which is essentially the same as NHWC = IHWO
    //unsigned int filterHeight = filterTensorInfo.GetShape()[1];
    //unsigned int filterWidth = filterTensorInfo.GetShape()[2];
    // [32, 3, 3, 3] on first conv

    auto filterOutCh = tp1->shape[0]; //
    int filterHeight = 1; //
    int filterWidth = 1;
    int filterInChannel = 0;

    auto filterRank = tp1->shape.size();
    if (filterRank == 2) {
        filterInChannel = tp1->shape[1];
    }
    else {
        auto filterHeight = tp1->shape[1]; //
        auto filterWidth = tp1->shape[2];
        auto filterInChannel = tp1->shape[3];
    }

    uint32_t padTop, padBottom, padLeft, padRight;
    padTop = padBottom = padLeft = padRight = 0; // all zero for fully connected with stride 1
 

    std::stringstream s;
    // converting FC to conv
    s << "conv" << subgraphIndex << "_" << operatorIndex;
    std::string nm = s.str();


    if (inputs.size() == 3)
    {
        auto tp2 = inputs[2];
    }
    else
    {
    }


    auto inputBufIndex = tp0->buffer;
    std::string inName;
    if (isRenamed(inputBufIndex)) {
        inName = getRename(inputBufIndex);
    }
    else {
        inName = tp0->name;
        if (inName == "input") {
            s.str("");
            if (opIndex > 0)
                s << "data" << opIndex;
            else s << "data";
            inName = s.str();
            tensorRename.push_back(std::pair<int, std::string>(inputBufIndex, inName));
            //[1,224,224,3] -> [1, 3, 224, 224]
            externDefs << inName << " = external(shape = [ 1, " <<
                tp0->shape[3] << ", " <<
                tp0->shape[1] << ", " <<
                tp0->shape[2] << "]);";
            std::string dat("data");
            //TODO check this tp0 ... probably need the data output vector
            auto q = tp0->quantization.get();
            inputScale = q->scale[0];

            outputFilterQuantizationInfo(externDefs, tp0, dat, 8, 0.0f);

        }
    }

    std::string filterName;
    std::string filterPath;
    std::string filterFolder;
    auto filterBufIndex = tp1->buffer;
    if (isRenamed(filterBufIndex)) {
        filterName = getRename(filterBufIndex);
        filterPath = getFilterPathRename(filterBufIndex);
        filterFolder = getFilterFolderRename(filterBufIndex);
    }
    else {
        s.str("");
        s << "filter" << opIndex;
        filterName = s.str();
        tensorRename.push_back(std::pair<int, std::string>(filterBufIndex, filterName));
        s.str("");
        s << "fc" << opIndex;
        filterFolder = s.str();
        filterFolderRename.push_back(std::pair<int, std::string>(filterBufIndex, filterFolder));
        s << "/filter";
        filterPath = s.str();
        filterPathRename.push_back(std::pair<int, std::string>(filterBufIndex, filterPath));

        //[1,224,224,3] -> [1, 3, 224, 224], so [och, h,w, ich] -> [och,ich,h,w] for filters
        // but in case of depthwise
        filterDefs << filterName << " = variable(shape = [" <<
            filterOutCh << ", " <<
            filterInChannel << ", " <<
            filterHeight << ", " <<
            filterWidth << "], label = '" << filterPath << "');";
        auto q = tp1->quantization.get();
        kernelScale = q->scale[0];

        outputFilterQuantizationInfo(filterDefs, tp1, filterName, 8, 0.0f);
        size_t filterSz = filterOutCh * filterHeight * filterWidth * filterInChannel;
        if (filterSz > maxFilterSz)
            maxFilterSz = filterSz;


    }

    std::string biasName = "";
    std::string biasPathName;
    std::string biasFolderName;
    if (inputs.size() == 3)
    {
        auto tp2 = inputs[2];
        auto biasBufIndex = tp2->buffer;
        if (isRenamed(biasBufIndex)) {
            biasName = getRename(biasBufIndex);
            biasPathName = getBiasPathRename(biasBufIndex);
            biasFolderName = getBiasFolderRename(biasBufIndex);
        }
        else {
            s.str("");
            s << "bias" << opIndex;
            biasName = s.str();
            tensorRename.push_back(std::pair<int, std::string>(biasBufIndex, biasName));
            s.str("");
            s << "fc" << opIndex;
            biasFolderName = s.str();
            biasFolderRename.push_back(std::pair<int, std::string>(biasBufIndex, biasFolderName));
            s << "/bias";
            biasPathName = s.str();
            biasPathRename.push_back(std::pair<int, std::string>(biasBufIndex, biasPathName));
            //[1,224,224,3] -> [1, 3, 224, 224]
            biasDefs << biasName << " = variable(shape = [1, " <<
                tp2->shape[0] << "], label = '" << biasPathName << "');";
            outputBiasQuantizationInfo(biasDefs, tp2, biasName, 32);
        }
    }
    std::string outName;
    auto tp3 = outputs[0];
    auto outBufIndex = tp3->buffer;
    if (isRenamed(outBufIndex)) {
        outName = getRename(outBufIndex);
    }
    else {
        s.str("");
        s << "fc" << opIndex;
        outName = s.str();
        tensorRename.push_back(std::pair<int, std::string>(outBufIndex, outName));
    }

    s.str("");
    std::string opName = "conv";
    s << "fc" << opIndex;
    s.str("");
    s << "fc" << opIndex << "/filter";
    filterPath = s.str();
    // we will create a std::pair with this index to rename the buffer

    netDefs << outName << " = conv(input = " << inName << ", filter = " << filterName <<
        ", bias = " << biasName << ", padding = [(" <<
        padTop << ", " <<
        padBottom << "), (" <<
        padLeft << ", " <<
        padRight << ")]";
    if (stride_h != 1) {
        netDefs << ", stride = [" << stride_w << ", " << stride_h << "]";
    }
    netDefs << ");";

    int outputHeight = 1;  // for first conv_dw this is [1,112,112,32]
                                        //unsigned int inputWidth = inputTensorInfo.GetShape()[2];
    int outputWidth = 1;
    int outputChannel = 0;

    auto outputRank = tp3->shape.size();
    if (outputRank == 2) {
        outputChannel = tp3->shape[1];
    }
    else {
        outputHeight = tp3->shape[1];  // for first conv_dw this is [1,112,112,32]
                                            //unsigned int inputWidth = inputTensorInfo.GetShape()[2];
        outputWidth = tp3->shape[2];
        outputChannel = tp3->shape[3];
    }


    netDefs << " # in = [1," << inputChannel << " ," << inputHeight << " ," << inputWidth <<
        " ], out= [1, " << outputChannel << " ," << outputHeight << " ," << outputWidth <<
        " ]";
    // TODO here needs to output also the downscale value
    // So probably need the third quantize definition? or else a non zero downscale param
    // TODO why are some of these scale sizes 0?  could be a schema difference.
    auto q = tp3->quantization.get();
    outputScale = (q->scale.size() == 0) ? 1.0 : q->scale[0];
    inputScale = (tp0->quantization->scale.size() == 0) ? 1.0 : tp0->quantization->scale[0];
    outputDownScale = inputScale * kernelScale / outputScale;
    outputFilterQuantizationInfo(netDefs, tp3, outName, 8, outputDownScale);

    // Here, let's tack on the input and output shapes for now.
    lastOutName = outName;
    opIndex++;
    return;
};



std::vector<OperatorParsingFunction>  m_ParserFunctions;
//using LayerBindingId = int;

static TensorIdRawPtrVector GetSubgraphInputs(const ModelPtr & model,
    size_t subgraphIndex)
{
    const auto & subGraphPtr = model->subgraphs[subgraphIndex];

    size_t inputCount = subGraphPtr->inputs.size();
    TensorIdRawPtrVector result(inputCount);
    for (size_t i = 0; i<inputCount; ++i)
    {
        uint32_t inputId = subGraphPtr->inputs[i];
        result[i] = std::make_pair(inputId, subGraphPtr->tensors[inputId].get());
    }
    return result;
}

static std::vector<int32_t>&  GetInputTensorIds(const ModelPtr& model,
    size_t subgraphIndex,
    size_t operatorIndex)
{
    const auto & subGraphPtr = model->subgraphs[subgraphIndex];
    const auto & operatorPtr = subGraphPtr->operators[operatorIndex];
    return operatorPtr->inputs;
}

static std::vector<int32_t>& GetOutputTensorIds(const ModelPtr& model,
    size_t subgraphIndex,
    size_t operatorIndex)
{
    const auto & subGraphPtr = model->subgraphs[subgraphIndex];
    const auto & operatorPtr = subGraphPtr->operators[operatorIndex];
    return operatorPtr->outputs;
}

// example usage: BufferRawPtr bufferPtr = GetBuffer(m_Model, inputs[0]->buffer);
BufferRawPtr GetBuffer(const ModelPtr& model, size_t bufferIndex)
{
    return model->buffers[bufferIndex].get();
}

static TensorRawPtrVector GetInputs(const ModelPtr & model,
    size_t subgraphIndex,
    size_t operatorIndex)
{

    const auto & subGraphPtr = model->subgraphs[subgraphIndex];
    const auto & operatorPtr = subGraphPtr->operators[operatorIndex];

    size_t inputCount = operatorPtr->inputs.size();
    TensorRawPtrVector result(inputCount);
    for (size_t i = 0; i<inputCount; ++i)
    {
        uint32_t inputId = operatorPtr->inputs[i];
        result[i] = subGraphPtr->tensors[inputId].get();
    }
    return result;
}
static TensorRawPtrVector  GetOutputs(const ModelPtr & model,
    size_t subgraphIndex,
    size_t operatorIndex)
{

    const auto & subGraphPtr = model->subgraphs[subgraphIndex];
    const auto & operatorPtr = subGraphPtr->operators[operatorIndex];

    size_t outputCount = operatorPtr->outputs.size();
    TensorRawPtrVector result(outputCount);
    for (size_t i = 0; i<outputCount; ++i)
    {
        uint32_t outputId = operatorPtr->outputs[i];
        result[i] = subGraphPtr->tensors[outputId].get();
    }
    return result;
}




static TensorIdRawPtrVector GetSubgraphOutputs(const ModelPtr & model,
    size_t subgraphIndex)
{
    const auto & subGraphPtr = model->subgraphs[subgraphIndex];

    size_t outputCount = subGraphPtr->outputs.size();
    TensorIdRawPtrVector result(outputCount);
    for (size_t i = 0; i<outputCount; ++i)
    {
        uint32_t outputId = subGraphPtr->outputs[i];
        result[i] = std::make_pair(outputId, subGraphPtr->tensors[outputId].get());
    }
    return result;
}





// This creates a jump table for the currently supported tflite operations.
// Anything not supported will go to ParseUnsupportedOperator.
void register_Operators()
{
    // register supported operators
    for (auto i = 0; i < tflite::BuiltinOperator_MAX + 1; i++) {
        m_ParserFunctions.push_back(&ParseUnsupportedOperator);
    }
    m_ParserFunctions[tflite::BuiltinOperator_AVERAGE_POOL_2D] = &ParseAveragePool2D;
    m_ParserFunctions[tflite::BuiltinOperator_CONV_2D] = &ParseConv2D;
    m_ParserFunctions[tflite::BuiltinOperator_ADD] = &ParseAdd;
    m_ParserFunctions[tflite::BuiltinOperator_DEPTHWISE_CONV_2D] = &ParseDepthwiseConv2D;
    m_ParserFunctions[tflite::BuiltinOperator_SOFTMAX] = &ParseSoftmax;
    m_ParserFunctions[tflite::BuiltinOperator_SQUEEZE] = &ParseSqueeze;
    m_ParserFunctions[tflite::BuiltinOperator_RESHAPE] = &ParseReshape;
    m_ParserFunctions[tflite::BuiltinOperator_MAX_POOL_2D] = &ParseMaxPool;
    m_ParserFunctions[tflite::BuiltinOperator_CONCATENATION] = &ParseConcat;
    m_ParserFunctions[tflite::BuiltinOperator_MEAN] = &ParseMean;
    m_ParserFunctions[tflite::BuiltinOperator_FULLY_CONNECTED] = &ParseFullyConnected;
    

}

// some of this is from armnn, but not needed in this converter
//struct QuantizationParametersT : public flatbuffers::NativeTable {
//    typedef QuantizationParameters TableType;
//    std::vector<float> min;
//    std::vector<float> max;
//    std::vector<float> scale;
//    std::vector<int64_t> zero_point;
//    QuantizationParametersT() {
//    }
//};

//enum TensorType {
//    TensorType_FLOAT32 = 0,
//    TensorType_FLOAT16 = 1,
//    TensorType_INT32 = 2,
//    TensorType_UINT8 = 3,
//    TensorType_INT64 = 4,
//    TensorType_STRING = 5,
//    TensorType_BOOL = 6,
//    TensorType_INT16 = 7,
//    TensorType_COMPLEX64 = 8,
//    TensorType_MIN = TensorType_FLOAT32,
//    TensorType_MAX = TensorType_COMPLEX64
//};


void write_nonTensorVals(
    std::string nm,
    size_t max_sz, float max,
    size_t min_sz, float min,
    size_t scale_sz, float scale,
    size_t zp_sz, int64_t zp,
    size_t shp_sz, int *shp,
    tflite::TensorType type,
    bool is_var,
    std::string nm_replace) {
    // so, do we need to do something here to pass through quantization values for the layer??
    // or is there enough info in the model so that we can just provide it at compile time??
    // do we need to write out the separate quantization file for the activation tensors?
    // the bias quantization has a learned range of values
    // and a downscale multiplier.  The bias downscale multiplier can be float or expressed as a 
    // integer multiply followed by a right shift and round.
    // for the weights and input there is also a zero offset value
    // The bias values are int32 (signed) and offset is considered 0
    // The multiplier is the scaling value
}



int getShapeSize(int shp_sz, int * shp) {
    int sz = 1;
    for (int i = 0; i < shp_sz; i++) {
        if (shp[i] > 0) {
            sz = sz * shp[i];
        }
    }
    return sz;
}

//NNEF specific header
struct Header {
    short magic;
    short version;
    int32_t len;
    int32_t rank;
    int32_t shape[8];
    int32_t  bits;
    short vendor_code; //00 = Khronos
    short alg_code;    //0x10 == linear_quant
    int32_t signed_unsigned; // 0 = unsigned, 1=signed
    float min;
    float max;
    float scale;
    uint8_t z_off;
    uint8_t downscale_shiftr;
    uint8_t min_sz;
    uint8_t max_sz;
    int32_t downscale_mpy;
    uint8_t scale_sz;
    uint8_t zp_sz;
    uint8_t ui8Param[6];
    uint8_t reserved[44];
} header;

typedef int*__restrict const tup_t;
typedef const int*__restrict const const_tup_t;

// Algorithm 2 of https://arxiv.org/ftp/arxiv/papers/1608/1608.00099.pdf
int tuple_to_index(const_tup_t tup, const_tup_t shape, unsigned int dimension) {
    int res = 0;
    int k;
    for (k = 0; k < dimension-1; ++k) {
        res += tup[k];
        res *= shape[k + 1];

    }
    res += tup[k];
    return res;
}

#ifndef WIN32

int dirExists(const char *path)
{
    struct stat info;

    if (stat(path, &info) != 0)
        return 0;
    else if (info.st_mode & S_IFDIR)
        return 1;
    else
        return 0;
}
int create_directories(const std::string path)
{

    std::string psub;
    int pos = 0;
    int len = path.length();
    while (pos < len ) {
        pos = path.find("/",  pos);
        if (pos == std::string::npos) {
            pos = len;
        }
        psub = path.substr(0, pos);
        if (!dirExists(psub.c_str())) {
            const int dir_err = mkdir(psub.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            if (-1 == dir_err)
            {
                printf("Error creating directory:%s\n", path.c_str());
                exit(1);
            }
        }
        pos++; // skip past path separator
    }
}

#endif
void write_TensorVals(
    std::string destRootFolder,
    std::string folder,
    std::string fnameReplace,
    std::string nm,
    size_t max_sz, float max,
    size_t min_sz, float min,
    size_t scale_sz, float scale,
    size_t zp_sz, int64_t zp,
    size_t shp_sz, int *shp,
    tflite::TensorType type,
    bool is_var,
    BufferRawPtr bufd,
    int bits,
    std::string nm_replace,
    uint32_t buf) {
    // destination folder
    std::string fname = destRootFolder + fnameReplace + ".dat";
    std::string foldernm = destRootFolder + folder; 
#ifdef WIN32
    fs::path folderPath(foldernm);
    if (!fs::exists(folderPath)) {
        fs::create_directories(folderPath);
    }
#else
    if (!dirExists(foldernm.c_str())) {
        create_directories(foldernm);
    }
#endif
    FILE *fp = fopen(fname.c_str(), "wb");
    //int sizeInWords = getShapeSize(shp_sz, shp);
    size_t bufSz = bufd->data.size();
    int headSz = sizeof(header);
    header.magic = 0xef4e;
    header.version = 0x0001;
    //header.vendor_code = (bits==8)? 0x1000: 0x0100; // Khronos linear quantized for uint8 weights, Khronos 32 bit int for bias
    header.signed_unsigned = (bits == 8) ? 0 : 1;
    header.len = bufSz & 0xffffffff;
    header.bits = bits;
    header.vendor_code = 0;
    header.alg_code = 0x10;
    for (int i = 0; i < 8; i++) header.shape[i] = 0;
    // ok, this needs to change to the new shape order format
    int shpr[4];
    int reorder[4];
    for (int i = 0; i < 4; i++) reorder[i] = i;
    if (bits == 8 && shp_sz == 4) {
        // NNEF expects NCHW insted of NHWC by tflite for activation
        // NNEF expects OIHW instead of OHWI for weights
        // for depthwise convert 1HWO to O1HW, for first dw [1,3,3,32]->[32,1,3,3]
        if (isFilterDw(buf)) {
            shpr[0] = shp[3]; reorder[0] = 3;
            shpr[1] = shp[0]; reorder[1] = 0;
            shpr[2] = shp[1]; reorder[2] = 1;
            shpr[3] = shp[2]; reorder[3] = 2;
        }
        else {
            // change OHWI to OIHW for conv and conv_sep
            shpr[0] = shp[0]; reorder[0] = 0;
            shpr[1] = shp[3]; reorder[1] = 3;
            shpr[2] = shp[1]; reorder[2] = 1;
            shpr[3] = shp[2]; reorder[3] = 2;
        }
    }
    else if (bits == 32 && shp_sz == 1) {
        // NNEF expects bias values [1,N]
        shpr[0] = 1;
        shpr[1] = shp[0];
        shp_sz = 2;
    }
    else {
        shpr[0] = shp[0];
        shpr[1] = shp[1];
        shpr[2] = shp[2];
        shpr[3] = shp[3];
    }
    // moved here since we have to resize the bias vectors
    header.rank = shp_sz;

    for (int i = 0; i < shp_sz; i++) header.shape[i] = shpr[i];

    header.max_sz = max_sz & 0xff;
    header.min_sz = min_sz & 0xff;
    header.scale_sz = scale_sz & 0xff;
    header.zp_sz = zp_sz;
    header.z_off = zp & 0xff;
    header.min = min;
    header.max = max;
    header.scale = scale;
    header.downscale_mpy = 0; // TODO calculate
    header.downscale_shiftr = 0; // TODO calculate
    header.bits = bits;
    for (int i = 0; i < 6; i++) header.ui8Param[i] = 0;
    for (int i = 0; i < 44; i++) header.reserved[i] = 0;
    size_t header_written = fwrite(&header, 1, sizeof(header), fp); //little endian 

   // We are converting from tflite to NNEF
    //kCR, // column-major matrix storage order. tflite standard.
    //    kRC, // row-major matrix storage order. TensorFlow default.
    //    kOHWI, // tflite standard for conv weights, caffe uses OIHW for weights
    //    kHWIO, // TensorFlow conv weights .. corresponds to intel mkldnn_hwio??
    //    k1HWO, // tflite standard for DepthwiseConv weights, caffe OIHW??
    //    kHWIM, // TensorFlow DepthwiseConv weights
    //    kNHWC, // TensorFlow activations, caffe uses NCHW, 
    //    kHWOI, // TensorFlow back-prop conv weights
    // NNEF uses BCHW for batch, channel, H, W tensor order where C is input channel
    // NNEF uses OIHW for output channel, in ch, h, w
    // Note that physical layout can be different
    // see https://github.com/intel/mkl-dnn/blob/master/include/mkldnn_types.h


    // insert here something to move the data to the right position
    uint8_t * from = (uint8_t *)&(bufd->data[0]);
    uint8_t * toBuf;
    int allocated = 0;
    if (bits == 8 && shp_sz == 4) {
        allocated = 1;
        toBuf = new uint8_t[bufSz];
        memset(toBuf, 0, bufSz);
        int from_off = 0;
        int to_off=0;
        // change OHWI to OIHW for conv and conv_sep
        // for depthwise convert 1HWO to O1HW, for first dw [1,3,3,32]->[32,1,3,3]

        int shpf[4];
        int tupFrom[4];
        int tupTo[4];
        int toIndex;
        for (int i = 0; i < 4; i++) shpf[i] = shp[i];
        for (int o = 0; o < shpf[0]; o++) {
            tupFrom[0] = o;
            for (int h = 0; h < shpf[1]; h++) {
                tupFrom[1] = h;
                for (int w = 0; w < shpf[2]; w++) {
                    tupFrom[2] = w;
                    for (int i = 0; i < shpf[3]; i++) {
                        tupFrom[3] = i;
                        // change OHWI to OIHW for conv and conv_sep
                        // for depthwise convert 1HWO to O1HW, so should be 1,3,3,32->32,1,3,3
                        //to_off = w + shpr[3] * h + i * shpr[2] * shpr[3] + o * shpr[1] * shpr[2] * shpr[3];
                        tupTo[0] = tupFrom[reorder[0]];
                        tupTo[1] = tupFrom[reorder[1]];
                        tupTo[2] = tupFrom[reorder[2]];
                        tupTo[3] = tupFrom[reorder[3]];
                        toIndex = tuple_to_index(tupTo, shpr, 4);

                        //TODO just here for debug remove this
                        if (toIndex >= bufSz) {
                            continue;
                        }
                        toBuf[toIndex] = from[from_off];

                         
                        from_off++;


                    }
                }
            }
        }
    }
    else {
        toBuf = from;
    }

    size_t num_written = fwrite(toBuf, 1, bufSz, fp); //little endian  
    fclose(fp);
    if (allocated) delete[] toBuf;
}

void CreateNetworkFromModel(const char* destFolder)
{

    bool failedToCreate = false;
    std::stringstream errors;
    std::string destRootFolder(destFolder);

    if (m_Model->subgraphs.size() != 1)
    {
        return;
    }

    size_t subgraphIndex = 0;
    for (SubGraphPtr const & subgraph : m_Model->subgraphs)
    {
        auto ten_size = subgraph->tensors.size();
       // m_SubgraphConnections.emplace_back(subgraph->tensors.size());

        // This goes through all the operations in order.
        // The inputs and outputs for each layer are determined.
        // The shapes are also determined, and  the names for bias, weight and other parameter files
        // The output shapes determined will be for NNEF, and so the weight and bias buffer
        // shape parameters will need to be updated, and the data itself will need to shuffle
        // for the weights.
        size_t operatorIndex = 0;
        for (OperatorPtr const & op : subgraph->operators)
        {
            //try
            {
                if (op->custom_options.size() > 0)
                {
                    return;
                 }

                auto const & opCodePtr = m_Model->operator_codes[op->opcode_index];
                auto builtinCode = opCodePtr->builtin_code;

                if (builtinCode > tflite::BuiltinOperator_MAX)
                {
                    return;
                 }

                // lookup and call the parser function
                auto & parserFunction = m_ParserFunctions[builtinCode];
                (*parserFunction)(subgraphIndex, operatorIndex);
            }
             ++operatorIndex;
        }

        // adding this to process all tensors
        // So, at this point we should have all the names for buffers and the new shapes
        //auto tens = subgraph->tensors;
        for (size_t i= 0; i<ten_size; i++){
            auto tp = subgraph->tensors[i].get();
            auto nm = tp->name.c_str();

            auto q = tp->quantization.get();
            float max = 0;
            float min = 0;
            float scale = 0;
            int64_t zp = 0;
            size_t max_sz = q->max.size();
            size_t min_sz = q->min.size();
            size_t scale_sz = q->scale.size();
            size_t zp_sz = q->zero_point.size();
            if (max_sz > 0)
                max = q->max[0];
            if (min_sz > 0)
                min = q->min[0];
            if (scale_sz > 0)
                scale = q->scale[0];
            if (zp_sz > 0)
                zp = q->zero_point[0];
            auto shp = tp->shape;
            size_t  shp_sz = shp.size();
            int32_t shpn[5] = { 0,0,0,0,0 };
            for (size_t j = 0; j < shp_sz && j < 5; j++) {
                shpn[j] = shp[j];
            }
            auto buf = tp->buffer; // this is a buffer index

            std::string nm_replace = isRenamed(buf) ? getRename(buf) : nm;

            auto type = tp->type; // TensorType_INT32 for bias, 
            auto is_var = tp->is_variable;

            // if it isn't a variable, then it is a tensor that holds data between operations.
            // If it is one of the between op vars, then we don't have to mutate it, but we do
            // need to correct the shape that is output, and we want to display the shape in the text
            // when it gets output.
            // Also, if not a variable, we don't have to create a file, but we may need to
            // use all this parameter data for the operation parameters, so how to handle that?
            int32_t biasv = 0;
            auto bufd = GetBuffer(m_Model, buf);
            // This size() is in uint_8 bytes, so int32 or float iterator will divide by 4
            if (bufd->data.size() == 0) {
                // input renamed "data" also comes here, which I guess can be an operation??
                // it currently shows up as the last op, but can't count on that.  Order seems random
                // see spec about graph.quant file, which can hold some quantization data.
                // We may want to store quant data in the graph.quant file, so that it can be available 
                // through the parser callbacks, rather than having to process it at runtime.
                parameterConstantFilenames.push_back(nm);
                write_nonTensorVals(tp->name, max_sz, max, min_sz, min, scale_sz, scale, zp_sz, zp,
                    shp_sz, shpn, type, is_var, nm_replace);

                //Relu6 goes here, although it has min and max values above
                // AveragePool goes here, although it has min and max values above
                // BiasAdd 1C goes here, but has min, max,scale values above and a shape size of 1001 ch
                // Reshape_1 comes here, but has min, max, scale values. the size of 2 just has the new shape
                // Input comes here, has scale and min and max between -1 and 0.992
                biasv = 0;
            }
            else if (type == tflite::TensorType_INT32) {

                // hmmm... what to do with the reshape node here?  It has some bias or scale value
                // before going to softmax.  Why wasn't this folded? Logits/SpatialSqueeze_shape
                // looks like operation just a pass-through, except for squeezing off the leading
                // 1s of the size 4  to the size 2 tensor.  This can be done in place.
                // do we already handle this in the softmax op?  should be a fused operation??
                int bits = 32;
                if (isRenamed(buf)) { //  this skips ones that have been folded, such as the reshape operation
                    // might  have to convert shpn to 2 here for bias values? and change shpn to precede by 1
                    std::string foldername = getBiasFolderRename(buf);
                    std::string fname = getBiasPathRename(buf);
                    write_TensorVals(destRootFolder,foldername, fname,tp->name, max_sz, max, min_sz, min, scale_sz, scale, zp_sz, zp,
                        shp_sz, shpn, type, is_var, bufd, bits, nm_replace, buf);
                    biasv = bufd->data[0] | (bufd->data[1] << 8) | (bufd->data[2] << 16) | bufd->data[3] << 24;
                    int32BiasFilenames.push_back(nm);
                }
                  // here for Conv2D_bias
                // also here for the SpatialSqueeze_shape for reshape, just two values int32
            }
            else if (type == tflite::TensorType_UINT8) {
                int bits = 8;
                std::string foldername = getFilterFolderRename(buf);
                std::string fname = getFilterPathRename(buf);

                write_TensorVals(destRootFolder,foldername, fname, tp->name, max_sz, max, min_sz, min, scale_sz, scale, zp_sz, zp,
                    shp_sz, shpn, type, is_var, bufd, bits, nm_replace,buf);
                uint8WeightFilenames.push_back(nm); // need to change shape
                // for example change to [512,1,3,3] from [1,3,3,512] and permute the data
                // need to ignore BiasAdd with uint8 type since it is folded
                // why does AvgPool have a uint8_t tensor?
                // hmmm... the conv2d_1c_1x1/weights/FakeQuantWithMinMaxVars includes 1001 batch size!x1024 bytes
                uint8_t bv = bufd->data[0];

            }
            else if (type == tflite::TensorType_FLOAT32) {
                //TODO, this conversion program currently only handles quantized uint8 and int32 bias tensors

            }
            else {
                return;
            }
        }

        ++subgraphIndex;
    }

    if (failedToCreate)
    {
    }
    return;
}


ModelPtr LoadModelFromBinary(const uint8_t * binaryContent, size_t len)
{
    if (binaryContent == nullptr)
    {
        return nullptr;
    }
    flatbuffers::Verifier verifier(binaryContent, len);
    if (verifier.VerifyBuffer<tflite::Model>() == false)
    {
        return nullptr;
    }
    m_Model =  tflite::UnPackModel(binaryContent);

    // has version = 3
    // 5 operator_codes
    // 1 subgraphs
    // 1 description (string)
    // 90 buffers, but buffer 0 is size 0 0x5a is buffers.size 80+10=90 = num buffer ptrs
    // 0 metadata buffers

    return nullptr;
    // return m_Model;
}

ModelPtr LoadModelFromFile(const char * fileName)
{
    if (fileName == nullptr)
    {
        return nullptr;
     }
     std::ifstream file(fileName, std::ios::binary);
    std::string fileContent((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return LoadModelFromBinary(reinterpret_cast<const uint8_t *>(fileContent.c_str()),
        fileContent.size());
}

inline char separator()
{
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}
void CreateNetworkFromBinaryFile(const char* graphFile, const char* destFolder)
{
    // makes assumption that the graphFile being converted is in a subdirectory
    // and ends in .tflite.  We're going to use this in the nnef graphname
    std::string graphName = graphFile;
    auto lastdot = graphName.find_last_of('.');
    if (lastdot == std::string::npos) return;

    auto lastSep = graphName.find_last_of(separator());
    if (lastSep != std::string::npos) {
        graphName = graphName.substr(lastSep + 1, lastdot - (lastSep + 1));
    }
    else {
        graphName = graphName.substr(0, lastdot-1);
    }
    // replace dot in graph name, which nnef may not like
    auto firstdot = graphName.find_last_of('.');
    if (lastdot != firstdot && firstdot != std::string::npos) {
        graphName.replace(firstdot, 1, "_");
    }

    graphHeader <<
        R"(version 1.0;
extension KHR_enable_fragment_definitions;
extension KHR_enable_operator_expressions;
fragment tflite_quantize( input: tensor<scalar>,  min: tensor<scalar>, max: tensor<scalar>, scale: tensor<scalar>, downscale: tensor<scalar>, offset: integer, bits: integer ) -> ( output: tensor<scalar> )
{
output = 0.0;
}
fragment tflite_bias_quantize( input: tensor<scalar>, scale: tensor<scalar>, bits: integer ) -> ( output: tensor<scalar> )
{
output = 0.0;
}
graph )" << graphName;


    LoadModelFromFile(graphFile);
    CreateNetworkFromModel(destFolder);


 
}


bool convert_to_nnef(const char* tfLiteFile, const char* destFolder)
{
 
    register_Operators();
    // this also creates the nnef compatible quantized binaries
    CreateNetworkFromBinaryFile(tfLiteFile, destFolder);

    // write to graph.nnef and graph.quant
    std::string outputFilename(destFolder);
    outputFilename +=   "graph.nnef";
    std::ofstream outFile;
    outFile.open(outputFilename);
    std::stringstream header;
    graphHeader << "( data ) -> ( " << lastOutName << " )\n{\n";
     
    outFile << graphHeader.rdbuf();
    outFile << externDefs.rdbuf();
    outFile << filterDefs.rdbuf();
    outFile << biasDefs.rdbuf();
    outFile << netDefs.rdbuf();
    outFile << "}\n"; // end of graph
    outFile.close();

    std::string quantFilename(destFolder);
    quantFilename += "graph.quant";
    outFile.open(quantFilename);

    outFile << quantDefs.rdbuf();
    outFile.close();

    return true;
}

int main(int argc, const char *argv[]) {
    std::string filename;
    std::string destFolder;
    if (argc < 3)
    {
        //std::cout << "Usage: tflite_to_nnef <input_quantized_filename.tflite> <desination_folder> " << std::endl;
        //std::cout << std::endl;
        printf("error, need args ... path to tflite file and output folder name\n");
        exit(1);

        // on windows like this:
        //filename = "C:\\mobilenetV2\\mobilenet_v2_1.0_224_quant\\mobilenet_v2_1.0_224_quant.tflite";
        //filename = "C:\\mobilenet\\mobilenet_v1_1.0_224_quant.tflite";
        //destFolder = "C:/MobilenetV1_tst/";
        //destFolder = "C:/MobilenetV2_tst/";
    }
    else {
        filename = argv[1];
        destFolder = argv[2];
    	{
	int len = destFolder.length();
	std::string lastch=destFolder.substr(len-1);
	if (lastch != "/") destFolder = destFolder + "/";
       
	}

    }

    convert_to_nnef(filename.c_str(),destFolder.c_str());
    return 0;
}
