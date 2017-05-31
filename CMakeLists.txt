cmake_minimum_required(VERSION 2.8)

#project(ukf C CXX)

find_package(Threads REQUIRED)
find_package(Boost 1.55 COMPONENTS program_options random filesystem system)

set(MUJOCO_INCLUDE_DIRS 
   #"${PROJECT_SOURCE_DIR}/source/mujoco"
   "/opt/mujoco140/include"
   #"/opt/mjfast/include"
)
set(MUJOCO_LIBRARY_DIRS
  #"/opt/mujoco/"
  "/opt/mujoco140/bin/"
	#"/opt/mjfast/bin/"
)

include_directories(
	"./"
	"/usr/local/include/eigen3"
  "/usr/include/eigen3"
	"/opt/local/include/eigen3"
	"/opt/local/include/"
  #"/opt/intel/mkl/include"
  "/usr/include/c++"
	${MUJOCO_INCLUDE_DIRS}
)


set(MKL_LIBRARIES
  "mkl_intel_lp64"
  "mkl_core"
  "mkl_gnu_thread"
  "m"
  "dl"
  #"pthread"
  #"mkl_sequential"
  )

set(MY_LINKER_LIBS "")

find_package(HDF5 COMPONENTS HL REQUIRED)
include_directories(SYSTEM ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIR})
#list(APPEND MY_LINKER_LIBS ${HDF5_LIBRARIES})
#list(APPEND MY_LINKER_LIBS hdf5)
#list(APPEND MY_LINKER_LIBS cnpy) # save / read npy numpy python files
#list(APPEND MY_LINKER_LIBS z) # crc32 check

list(APPEND MY_LINKER_LIBS ${Boost_LIBRARIES}) # boost

link_directories(
  #"/opt/intel/mkl/lib/intel64/"
	${MUJOCO_LIBRARY_DIRS}
  ${HDF5_LIBRARY_DIRS}
)


#option(MUJOCO_SIMULATOR_PARALLEL "OpenMP for parallel Mujoco." on)
option(USE_EIGEN_MKL "Use MKL for Eigen backend." off)
if (USE_EIGEN_MKL)
  add_definitions(-DUSE_EIGEN_MKL)
endif()

#option(ENABLE_SOCKET_STREAMING "Set Mujoco Viewer to support data streaming" off)

if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall ")
endif()

if (APPLE)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_THREAD_LIBS_INIT} -std=c++11")
   set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/mac_bin)
   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/mac_bin)
	 set(MUJOCO_LIBRARIES "glfw" "mujoco")
else()
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
   set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
   set(MUJOCO_LIBRARIES "glfw" "mujoco" "stdc++" "glew" "GL")
endif()

# parallel mujoco option from top-level cmake
#if(MUJOCO_SIMULATOR_PARALLEL)
#   find_package(OpenMP REQUIRED)
#   if (OPENMP_FOUND)
#      add_definitions(-DMUJOCO_SIMULATOR_PARALLEL)
#      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#   endif()
#endif(MUJOCO_SIMULATOR_PARALLEL)

if (NOT APPLE)
   # Darwin Code
   subdirs(darwin_hw)
   # Viewer Rendering
   add_executable(viewer "viewer.cpp" "ukf.cpp" "util_func.cpp" "lodepng.cpp" "pf.cpp" "kNN.cpp" "kNNPF.cpp")
   #target_link_libraries(viewer ${MUJOCO_LIBRARIES} )
   if(USE_EIGEN_MKL)
     target_link_libraries(viewer ${MUJOCO_LIBRARIES} ${MY_LINKER_LIBS} ${MKL_LIBRARIES}
       darwin_hw hdf5 hdf5_cpp)
   else()
     target_link_libraries(viewer ${MUJOCO_LIBRARIES} ${MY_LINKER_LIBS} darwin_hw
       ${DARWIN_INCLUDES} hdf5 hdf5_cpp)
   endif()
   #target_include_directories(viewer PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
   add_executable(test_est "test_est.cpp" "ukf.cpp" "darwin_hw/drwn_walker.cpp" "util_func.cpp" 
     "darwin_hw/Vector.cpp" "darwin_hw/Matrix.cpp" "darwin_hw/MX28.cpp"
     "darwin_hw/Point.cpp" "darwin_hw/JointData.cpp" )
   target_link_libraries(test_est ${MUJOCO_LIBRARIES} ${MY_LINKER_LIBS} )

   # Viewer Compare
   add_executable(viewCompare "compare_viewer.cpp" "ukf.cpp" "util_func.cpp" "lodepng.cpp" "pf.cpp" "kNN.cpp" "kNNPF.cpp")
   if(USE_EIGEN_MKL)
     target_link_libraries(viewCompare ${MUJOCO_LIBRARIES} ${MY_LINKER_LIBS} ${MKL_LIBRARIES}
       darwin_hw hdf5 hdf5_cpp)
   else()
     target_link_libraries(viewCompare ${MUJOCO_LIBRARIES} ${MY_LINKER_LIBS} darwin_hw
       ${DARWIN_INCLUDES} hdf5 hdf5_cpp)
   endif()

else()
  add_library(sim_only "darwin_hw/drwn_walker.cpp" "util_func.cpp" 
    "darwin_hw/Vector.cpp" "darwin_hw/Matrix.cpp" "darwin_hw/MX28.cpp"
    "darwin_hw/Point.cpp" "darwin_hw/JointData.cpp" )
  add_executable(viewer "viewer.cpp" "ukf.cpp" "util_func.cpp" "lodepng.cpp" "pf.cpp" "kNN.cpp" "kNNPF.cpp")
  target_link_libraries(viewer sim_only ${MUJOCO_LIBRARIES} ${MY_LINKER_LIBS} )
endif()

#add_executable(speed_test "multithread.cpp" )
#target_link_libraries(speed_test ${MUJOCO_LIBRARIES} )

add_executable(sim "simulate.cpp" )
target_link_libraries(sim ${MUJOCO_LIBRARIES} )

#add_executable(mc "mcsim.cpp" )
#target_link_libraries(mc ${MUJOCO_LIBRARIES} ${MY_LINKER_LIBS} )

add_executable(testekf "testEKF.cpp" )
target_link_libraries(testekf ${MUJOCO_LIBRARIES} ${MY_LINKER_LIBS} )

add_executable(makekNN "make_kNN.cpp" "util_func.cpp" "kNN.cpp")
target_link_libraries(makekNN ${MUJOCO_LIBRARIES} ${MY_LINKER_LIBS} hdf5 hdf5_cpp )


