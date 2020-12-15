cmake_minimum_required(VERSION 2.8)
project(reduccionMPI)
find_package( OpenCV REQUIRED )
find_package(MPI) #make it REQUIRED, if you want

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( reduccionMPI reduccionMPI.cc)

target_link_libraries( reduccionMPI ${MPI_LIBRARIES} ${OpenCV_LIBS}  -pthread -lm )


