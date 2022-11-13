cmake_minimum_required(VERSION 3.13)

# project start here
project(onnx_interp C CXX)

include(FetchContent)
set(THIRD_PARTY ${CMAKE_SOURCE_DIR}/3rd_party)
set(ONNXRUNTIME_ROOTDIR ${THIRD_PARTY}/onnxruntime)

if(MSVC)
        set(URL_ONNXRUNTIME https://github.com/microsoft/onnxruntime/releases/download/v1.12.0/onnxruntime-win-x64-1.12.0.zip)
        string(APPEND CMAKE_CXX_FLAGS " /W4")
        set(ONNXRUNTIME_LIBRARIES
    	        ${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime.lib
    	        ${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime_providers_shared.lib
    	        )
elseif(UNIX AND NOT APPLE)
        set(URL_ONNXRUNTIME https://github.com/microsoft/onnxruntime/releases/download/v1.12.0/onnxruntime-linux-x64-1.12.0.tgz)
        string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
        string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
	set(ONNXRUNTIME_LIBRARIES
                ${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so
                )
	set(CMAKE_INSTALL_RPATH "\$ORIGIN")
else()
	message(FATAL_ERROR "unknown target")
endif()

#onnxruntime providers
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
if(onnxruntime_USE_CUDA)
        add_definitions(-DUSE_CUDA)
endif()

# add ONNX Runtime
if(NOT EXISTS ${ONNXRUNTIME_ROOTDIR})
        message("** Download onnxruntime.")
        FetchContent_Declare(onnxruntime
                URL ${URL_ONNXRUNTIME}
                SOURCE_DIR ${ONNXRUNTIME_ROOTDIR}
                )
        FetchContent_MakeAvailable(onnxruntime)
endif()

include_directories(
	${ONNXRUNTIME_ROOTDIR}/include
	${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session
	)
link_libraries(
	${ONNXRUNTIME_LIBRARIES}
	)

if(MSVC)
	file(GLOB NNFW_DLLS "${ONNXRUNTIME_ROOTDIR}/lib/*.dll")
	message("${NNFW_DLLS}")
elseif(UNIX AND NOT APPLE)
	file(GLOB NNFW_DLLS "${ONNXRUNTIME_ROOTDIR}/lib/*.so*")
	message("${NNFW_DLLS}")
endif()


# add Nlohmann JSON
set(URL_NLOHMANN_JSON "https://github.com/nlohmann/json/releases/download/v3.7.3/include.zip")
set(NLOHMANN_JSON_ROOTDIR ${THIRD_PARTY}/nlohmann_json)

if(NOT EXISTS ${NLOHMANN_JSON_ROOTDIR})
        message("** Download nlohmann_json.")
        FetchContent_Declare(nlohmann_json
                URL ${URL_NLOHMANN_JSON}
                SOURCE_DIR ${NLOHMANN_JSON_ROOTDIR}
                )
        FetchContent_MakeAvailable(nlohmann_json)
endif()

include_directories(${NLOHMANN_JSON_ROOTDIR}/include)

# my own libraries
set(GETOPT 
	src/getopt/getopt.c
	src/getopt/getopt_long.c
	)
add_library(interp
	STATIC
	src/tiny_ml.cpp
	src/tensor_spec.cpp
	src/io_port.cpp
	src/nonmaxsuppression.cpp
	${GETOPT}
	)

# main
add_executable(onnx_interp
	src/main.cpp
	src/onnx_interp.cpp
	)
target_link_libraries(onnx_interp
	interp
	)

# installation
install(TARGETS onnx_interp
	RUNTIME
	DESTINATION $ENV{MIX_APP_PATH}/priv
	)
install(TARGETS onnx_interp
	RUNTIME
	DESTINATION ${CMAKE_SOURCE_DIR}/priv
	)

if(NNFW_DLLS)
	add_custom_command(TARGET onnx_interp
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory $ENV{MIX_APP_PATH}/priv
		COMMAND ${CMAKE_COMMAND} -E copy_if_different ${NNFW_DLLS} $ENV{MIX_APP_PATH}/priv
		)
	add_custom_command(TARGET onnx_interp
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/priv
		COMMAND ${CMAKE_COMMAND} -E copy_if_different ${NNFW_DLLS} ${CMAKE_SOURCE_DIR}/priv
		)
endif()
