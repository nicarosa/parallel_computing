cmake_minimum_required(VERSION 2.8)
project( Reduction LANGUAGES CXX CUDA)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(/usr/local/cuda-10.0/samples/common/inc/)
set_source_files_properties(reduction.cu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable( reduction reduction.cu)

target_link_libraries( reduction ${OpenCV_LIBS} -lm )


