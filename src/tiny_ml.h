/***  File Header  ************************************************************/
/**
* @file tiny_ml.h
*
* system setting - used throughout the system
* @author   Shozo Fukuda
* @date     create Sun Feb 13 15:17:40 JST 2022
* System    MINGW64/Windows 10<br>
*
*******************************************************************************/
#ifndef _TINY_ML_H
#define _TINY_ML_H

#include <iostream>
#include <string>
#include <vector>
#include <functional>

#include <chrono>
namespace chrono = std::chrono;

#include "nlohmann/json.hpp"
using json = nlohmann::json;

#ifndef __GNUC__
#  define __attribute__(x)
#endif

/***  Class Header  *******************************************************}}}*/
/**
* Abstruct Tiny ML Interpreter
* @par DESCRIPTION
*   Interface of Tiny ML Interpreter
*
**/
/**************************************************************************{{{*/
class TinyMLInterp{
//LIFECYCLE:
public:
    TinyMLInterp() {}
    virtual ~TinyMLInterp() {}

//ACTION:
public:
	virtual void info(json& res) = 0;
	virtual int set_input_tensor(unsigned int index, const uint8_t* data, int size) = 0;
	virtual int set_input_tensor(unsigned int index, const uint8_t* data, int size, std::function<float(uint8_t)> conv) = 0;
	virtual bool invoke() = 0;
	virtual std::string get_output_tensor(unsigned int index) = 0;

//INQUIRY:
public:
    size_t InputCount()  { return mInputCount;  }
    size_t OutputCount() { return mOutputCount; }

//ATTRIBUTE:
protected:
    size_t mInputCount;
    size_t mOutputCount;
};

/**************************************************************************}}}**
* system information
***************************************************************************{{{*/
#define NUM_LAP	10

struct SysInfo {
    std::string    mExe;       // path of this executable
    std::string    mModelPath; // path of Tflite Model
    std::string    mLabelPath; // path of Class Labels
    unsigned long mDiag;       // diagnosis mode
    int            mNumThread;  // number of thread

    TinyMLInterp* mInterp{nullptr};

    std::vector<std::string> mLabel;
    size_t mNumClass;

    // i/o method
    int (*mRcv)(std::string& cmd_line);
    int (*mSnd)(std::string result);

    std::string label(int id) {
        return (id < mLabel.size()) ? mLabel[id] : std::to_string(id);
    }
    
    // stop watch
	chrono::steady_clock::time_point mWatchStart;
	chrono::milliseconds mLap[NUM_LAP];

    void reset_lap() {
    	for (int i = 0; i < NUM_LAP; i++) { mLap[i] = chrono::milliseconds(0); }
    }
    void start_watch() {
    	reset_lap();
    	mWatchStart = chrono::steady_clock::now();
    }
    void lap(int index) {
    	chrono::steady_clock::time_point now = chrono::steady_clock::now();
    	mLap[index] = chrono::duration_cast<chrono::milliseconds>(now - mWatchStart);
    	mWatchStart = now;
    }
};

#define LAP_INPUT()		lap(0)
#define LAP_EXEC()		lap(1)
#define LAP_OUTPUT()	lap(2)

extern SysInfo gSys;

/**************************************************************************}}}**
* i/o functions
***************************************************************************{{{*/
int rcv_packet_port(std::string& cmd_line);
int snd_packet_port(std::string result);

/**************************************************************************}}}**
* service call functions
***************************************************************************{{{*/
void interp(std::string& model, std::string& labels);
void init_interp(SysInfo& sys, std::string& model);

#endif /* _TINY_ML_H */
