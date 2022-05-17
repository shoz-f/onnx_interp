# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.13)

################################################################################
# project start here
project(onnxruntime_samples C CXX)

include(FetchContent)

if (WIN32)
    string(APPEND CMAKE_CXX_FLAGS " /W4")
    set(URL_ONNXRUNTIME https://github.com/microsoft/onnxruntime/releases/download/v1.11.0/onnxruntime-win-x64-1.11.0.zip)
else()
    string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
    string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
endif()

#onnxruntime providers
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
if(onnxruntime_USE_CUDA)
  add_definitions(-DUSE_CUDA)
endif()


# add ONNX Runtime
set(ONNXRUNTIME_ROOTDIR ${CMAKE_SOURCE_DIR}/3rd_party/onnxruntime)

if(NOT EXISTS ${ONNXRUNTIME_ROOTDIR})
message("** Download onnxruntime.")
FetchContent_Declare(onnxruntime
	URL ${URL_ONNXRUNTIME}
	SOURCE_DIR ${ONNXRUNTIME_ROOTDIR}
	)
else()
FetchContent_Declare(onnxruntime
	SOURCE_DIR ${ONNXRUNTIME_ROOTDIR}
)
endif()
FetchContent_MakeAvailable(onnxruntime)

include_directories(
	${ONNXRUNTIME_ROOTDIR}/include
	${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session
	)
link_directories(
	${ONNXRUNTIME_ROOTDIR}/lib
	)

# add Nlohmann JSON
if(NOT EXISTS ${CMAKE_SOURCE_DIR}/3rd_party/nlohmann_json)
message("** Download nlohmann_json.")
FetchContent_Declare(nlohmann_json
	GIT_REPOSITORY https://github.com/nlohmann/json.git
	GIT_PROGRESS   TRUE
	SOURCE_DIR "${CMAKE_SOURCE_DIR}/3rd_party/nlohmann_json"
	)
else()
FetchContent_Declare(nlohmann_json
	SOURCE_DIR "${CMAKE_SOURCE_DIR}/3rd_party/nlohmann_json"
)
endif()
FetchContent_MakeAvailable(nlohmann_json)

# my own libraries
add_library(getopt
	STATIC
	src/getopt/getopt.c
	src/getopt/getopt_long.c
	)

# main
add_executable(onnx_interp
	src/main.cpp
	src/tiny_ml.cpp
	src/onnx_interp.cpp
	src/io_port.cpp
	src/nonmaxsuppression.cpp
	)
target_link_libraries(onnx_interp
	onnxruntime
	onnxruntime_providers_shared
	getopt
	nlohmann_json
	)

# installation
install(TARGETS onnx_interp
	RUNTIME
	DESTINATION $ENV{MIX_APP_PATH}/priv
	)
install(DIRECTORY ${ONNXRUNTIME_ROOTDIR}/lib/
	DESTINATION $ENV{MIX_APP_PATH}/priv
	FILES_MATCHING PATTERN "*.dll"
	)

install(TARGETS onnx_interp
	RUNTIME
	DESTINATION ${CMAKE_SOURCE_DIR}/priv
	)
install(DIRECTORY ${ONNXRUNTIME_ROOTDIR}/lib/
	DESTINATION ${CMAKE_SOURCE_DIR}/priv
	FILES_MATCHING PATTERN "*.dll"
	)
