cmake_minimum_required(VERSION 2.8)

#project(ukf C CXX)

#find_package(CUDA 7.0)
#find_package(Ceres)
find_package(Threads REQUIRED)

#add_definitions(${Caffe_DEFINITIONS})
#set(Caffe_INCLUDE_DIRS 
#	"/opt/caffe/include"
#	"/usr/local/cuda/include"
#)
#set(Caffe_LIBRARY_DIRS
#	"/opt/caffe/lib"
#	"/usr/local/cuda/lib64"
#	"/opt/OpenBLAS/lib"
#)
#set(Caffe_LIBRARIES
#	"cublas"
#	"caffe"
#    #"openblas"
#	"glog"
#)

#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(MUJOCO_INCLUDE_DIRS 
   #"${PROJECT_SOURCE_DIR}/source/mujoco"
   #"/opt/mujoco122"
   "/opt/mujoco130/include"
)

set(MUJOCO_LIBRARY_DIRS
  #"/opt/mujoco/"
	"/opt/mujoco130/bin/"
)

set(MUJOCO_LIBRARIES
	"glfw"
	"mujoco"
)

include_directories(
	"./"
	"/usr/local/include/eigen3"
  "/opt/intel/mkl/include"
	${MUJOCO_INCLUDE_DIRS}
)
link_directories(
  "/opt/intel/mkl/lib/intel64/"
	${MUJOCO_LIBRARY_DIRS}
)

set(MKL_LIBRARIES
  "mkl_intel_lp64"
  "mkl_core"
  "mkl_gnu_thread"
  "m"
  "dl"
  "pthread"
  #"mkl_sequential"
  )

#link_libraries(
#    ${Boost_LIBRARIES}
#    ${PYTHON_LIBRARIES}
#)

option(MUJOCO_SIMULATOR_PARALLEL "OpenMP for parallel Mujoco." on)
option(USE_EIGEN_MKL "Use MKL for Eigen backend." off)
if (USE_EIGEN_MKL)
  add_definitions(-DUSE_EIGEN_MKL)
endif()
#option(ENABLE_SOCKET_STREAMING "Set Mujoco Viewer to support data streaming" off)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_THREAD_LIBS_INIT} -std=gnu++11 -fopenmp -O2")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# parallel mujoco option from top-level cmake
#if(MUJOCO_SIMULATOR_PARALLEL)
#   find_package(OpenMP REQUIRED)
#   if (OPENMP_FOUND)
#      add_definitions(-DMUJOCO_SIMULATOR_PARALLEL)
#      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#   endif()
#endif(MUJOCO_SIMULATOR_PARALLEL)

# Darwin Code
subdirs(darwin_hw)

# Viewer Rendering
add_executable(viewer "viewer.cpp" )
#target_link_libraries(viewer ${MUJOCO_LIBRARIES} )
if(USE_EIGEN_MKL)
  target_link_libraries(viewer ${MUJOCO_LIBRARIES} darwin_hw mkl_intel_lp64 mkl_core
    mkl_gnu_thread pthread m dl)
else()
  target_link_libraries(viewer ${MUJOCO_LIBRARIES} darwin_hw )
endif()
#target_include_directories(viewer PUBLIC ${CMAKE_CURRENT_BINARY_DIR})


