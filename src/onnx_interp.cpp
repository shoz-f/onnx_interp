/***  File Header  ************************************************************/
/**
* onnx_interp.cpp
*
* Elixir/Erlang Port ext. of tensor flow lite
* @author   Shozo Fukuda
* @date     create Sat Sep 26 06:26:30 JST 2020
* System    Visual C++/Windows 10<br>
*
**/
/**************************************************************************{{{*/

#include <stdio.h>
#include "onnx_interp.h"

/*--- CONSTANT ---*/
const std::string _dtype[] = {
    "UNDEFINED",
    "FLOAT",        // maps to c type float
    "UINT8",        // maps to c type uint8_t
    "INT8",         // maps to c type int8_t
    "UINT16",       // maps to c type uint16_t
    "INT16",        // maps to c type int16_t
    "INT32",        // maps to c type int32_t
    "INT64",        // maps to c type int64_t
    "STRING",       // maps to c++ type std::string
    "BOOL",
    "FLOAT16",
    "DOUBLE",       // maps to c type double
    "UINT32",       // maps to c type uint32_t
    "UINT64",       // maps to c type uint64_t
    "COMPLEX64",    // complex with float32 real and imaginary components
    "COMPLEX128",   // complex with float64 real and imaginary components
    "BFLOAT16"      // Non-IEEE floating-point format based on IEEE754 single-precision
};

/***  Module Header  ******************************************************}}}*/
/**
* query dimension of input tensor
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
size_t
get_tensor_size(
Ort::Value& value)
{
	size_t size;

	auto tensor_info = value.GetTensorTypeAndShapeInfo();

	ONNXTensorElementDataType type = tensor_info.GetElementType();

	switch (type) {
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
		return 0;

	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
		size = 8;
		break;

	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
		size = 4;
		break;

	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
		size = 2;
		break;
	
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
		size = 1;
		break;

	default:
		return 0;
	}

	size_t shape_len = tensor_info.GetDimensionsCount();
	int64_t* shape = new int64_t[shape_len];
	tensor_info.GetDimensions(shape, shape_len);
	for (int j = 0; j < shape_len; j++) {
		if (shape[j] == -1) { shape[j] = 1; }

		size *= shape[j];
	}

	delete [] shape;

	return size;
}

/***  Method Header  ******************************************************}}}*/
/**
* constructor
* @par DESCRIPTION
*   construct an instance.
**/
/**************************************************************************{{{*/
OnnxInterp::OnnxInterp(std::string onnx_model)
{
	Ort::AllocatorWithDefaultOptions _ort_alloc;
    Ort::SessionOptions session_options;

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    std::wstring widestr = std::wstring(onnx_model.begin(), onnx_model.end());
    mSession = Ort::Session(mEnv, widestr.c_str(), session_options);

    mInputCount = mSession.GetInputCount();
    mInputNames = new char*[mInputCount];
    for (int i = 0; i < mInputCount; i++) {
        mInputNames[i] = mSession.GetInputName(i, _ort_alloc);
        
        auto tensor_info = mSession.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        
        ONNXTensorElementDataType type = tensor_info.GetElementType();

        size_t shape_len = tensor_info.GetDimensionsCount();
        int64_t* shape = new int64_t[shape_len];
        tensor_info.GetDimensions(shape, shape_len);
        for (int j = 0; j < shape_len; j++) {
        	if (shape[j] == -1) { shape[j] = 1; }
        }

		mInput.emplace_back(Ort::Value::CreateTensor(_ort_alloc, shape, shape_len, type));
        
        delete [] shape;
    }

    mOutputCount = mSession.GetOutputCount();
    mOutputNames = new char*[mOutputCount];
    for (int i = 0; i < mOutputCount; i++) {
        mOutputNames[i] = mSession.GetOutputName(i, _ort_alloc);

        auto tensor_info = mSession.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
        
        ONNXTensorElementDataType type = tensor_info.GetElementType();

        size_t shape_len = tensor_info.GetDimensionsCount();
        int64_t* shape = new int64_t[shape_len];
        tensor_info.GetDimensions(shape, shape_len);
        for (int j = 0; j < shape_len; j++) {
        	if (shape[j] == -1) { shape[j] = 1; }
        }
        
        mOutput.push_back(Ort::Value::CreateTensor(_ort_alloc, shape, shape_len, type));
        
        delete [] shape;
    }
}

/***  Method Header  ******************************************************}}}*/
/**
* destructor
* @par DESCRIPTION
*   delate an instance.
**/
/**************************************************************************{{{*/
OnnxInterp::~OnnxInterp()
{
	Ort::AllocatorWithDefaultOptions _ort_alloc;

    for (int i = 0; i < mInputCount; i++) {
        _ort_alloc.Free(mInputNames[i]);
    }
    delete [] mInputNames;

    for (int i = 0; i < mOutputCount; i++) {
        _ort_alloc.Free(mOutputNames[i]);
    }
    delete [] mOutputNames;
}

/***  Module Header  ******************************************************}}}*/
/**
* initialize interpreter
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
void init_interp(SysInfo& sys, std::string& onnx_model)
{
    // load tensor flow lite model
//    CHECK(g_ort2->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnx_interp", &sys.mEnv));
//
//    CHECK(g_ort2->CreateSessionOptions(&sys.mSessionOptions));
//    CHECK(g_ort2->SetIntraOpNumThreads(sys.mSessionOptions, sys.mNumThread));
//    CHECK(g_ort2->SetSessionGraphOptimizationLevel(sys.mSessionOptions, ORT_ENABLE_BASIC));
//
//    std::wstring widestr = std::wstring(onnx_model.begin(), onnx_model.end());
//    CHECK(g_ort2->CreateSession(sys.mEnv, widestr.c_str(), sys.mSessionOptions, &sys.mSession));

//    if (sys.mInterpreter->AllocateTensors() != kTfLiteOk) {
//        std::cerr << "error: AllocateTensors()\n";
//        exit(1);
//    }

    sys.mInterp = new OnnxInterp(onnx_model);
}

/***  Module Header  ******************************************************}}}*/
/**
* query dimension of input tensor
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
void
OnnxInterp::info(json& res)
{
    for (int index = 0; index < mInputCount; index++) {
        json onnx_tensor;

        onnx_tensor["index"] = index;
        onnx_tensor["name"]  = mInputNames[index];

        auto tensor_info = mSession.GetInputTypeInfo(index).GetTensorTypeAndShapeInfo();

        onnx_tensor["type"] = _dtype[tensor_info.GetElementType()];

        for (const auto& n : tensor_info.GetShape()) {
            if (n == -1) {
                onnx_tensor["dims"].push_back("none");
            }
            else {
                onnx_tensor["dims"].push_back(n);
            }
        }

        res["inputs"].push_back(onnx_tensor);
    }

    for (int index = 0; index < mOutputCount; index++) {
        json onnx_tensor;

        onnx_tensor["index"] = index;
        onnx_tensor["name"]  = mOutputNames[index];

        auto tensor_info = mSession.GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo();

        onnx_tensor["type"] = _dtype[tensor_info.GetElementType()];

        for (const auto& n : tensor_info.GetShape()) {
            if (n == -1) {
                onnx_tensor["dims"].push_back("none");
            }
            else {
                onnx_tensor["dims"].push_back(n);
            }
        }

        res["outputs"].push_back(onnx_tensor);
    }
}

/***  Module Header  ******************************************************}}}*/
/**
* set input tensor
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
int
OnnxInterp::set_input_tensor(unsigned int index, const uint8_t* data, int size)
{
 //   if (size != mInput[index].GetStringTensorDataLength()) {
 //       return -2;
 //   }

    memcpy(mInput[index].GetTensorMutableData<uint8_t>(), data, size);

    return size;
}

/***  Module Header  ******************************************************}}}*/
/**
* set input tensor
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
int
OnnxInterp::set_input_tensor(unsigned int index, const uint8_t* data, int size, std::function<float(uint8_t)> conv)
{
    if (size != mInput[index].GetStringTensorDataLength()/sizeof(float)) {
        return -2;
    }

    float* dst = mInput[index].GetTensorMutableData<float>();
    const uint8_t* src = data;
    for (int i = 0; i < size; i++) {
        *dst++ = conv(*src++);
    }

    return size;
}

/***  Module Header  ******************************************************}}}*/
/**
* execute inference
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
bool
OnnxInterp::invoke()
{
    mOutput = mSession.Run(Ort::RunOptions{nullptr}, mInputNames, mInput.data(), mInputCount, mOutputNames, mOutputCount);
    return true;
}

/***  Module Header  ******************************************************}}}*/
/**
* get result tensor
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
std::string
OnnxInterp::get_output_tensor(unsigned int index)
{
//    return std::string(mOutput[index].GetTensorData<char>(), mOutput[index].GetStringTensorDataLength());
    return std::string(mOutput[index].GetTensorData<char>(), get_tensor_size(mOutput[index]));
}

#if 0
/***  Module Header  ******************************************************}}}*/
/**
* execute inference in session mode
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
std::string
run(SysInfo& sys, const void* args)
{
    // set input tensors
    struct Prms {
        unsigned int  count;
        unsigned char data[0];
    } __attribute__((packed));
    const Prms* prms = reinterpret_cast<const Prms*>(args);

    sys.start_watch();

    const unsigned char* ptr = prms->data;
    for (int i = 0; i < prms->count; i++) {
        int next = set_itensor(sys, ptr);
        if (next < 0) {
            // error about input tensors: error_code {-1..-3}
            return std::string(reinterpret_cast<char*>(&next), sizeof(next));
        }

        ptr += next;
    }

    sys.LAP_INPUT();

    // invoke
    mSession.Run(, mInputNames, , mInputCount, mOutputNames, , mOutputCount);
    int status = sys.mInterpreter->Invoke();
    if (status != kTfLiteOk) {
        // error about invoke: error_code {-11..}
        status = -(10 + status);
        return std::string(reinterpret_cast<char*>(&status), sizeof(status));
    }

    sys.LAP_EXEC();

    // get output tensors  <<count::little-integer-32, size::little-integer-32, bin::binary-size(size), ..>>
    int count = sys.mInterpreter->outputs().size();
    std::string output(reinterpret_cast<char*>(&count), sizeof(count));

    for (int index = 0; index < count; index++) {
        TfLiteTensor* otensor = sys.mInterpreter->output_tensor(index);
        int size = otensor->bytes;
        output += std::string(reinterpret_cast<char*>(&size), sizeof(size))
               +  std::string(otensor->data.raw, otensor->bytes);
    }

    sys.LAP_OUTPUT();

    return output;
}
#endif
/*** onnx_interp.cc ********************************************************}}}*/
