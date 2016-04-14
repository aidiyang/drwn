# protobuf compilation
#file(GLOB ProtoFiles "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")
#PROTOBUF_GENERATE_CPP(ProtoSources ProtoHeaders ${ProtoFiles})

file(GLOB DARWIN_SOURCES "*.cpp" )
list(REMOVE_ITEM DARWIN_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/test_contacts.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_darwin.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/loop_time.cpp )
if (WIN32)
  list(REMOVE_ITEM DARWIN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/LinuxCM730.cpp )
else ()
  list(REMOVE_ITEM DARWIN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/WindowsCM730.cpp )
endif (WIN32)

message(STATUS ${DARWIN_SOURCES})

###################################
# Phidgets include dir and Library
#find_path(PHIDGETS_INCLUDE_DIR
#  NAMES
#  phidget21.h
#  PATHS
#  "/usr/local/include"
#  "/usr/include"
#  )
#find_library(PHIDGETS_LIBRARIES
#  NAMES
#  phidget21
#  PATHS
#  "/usr/local/lib"
#  )
#
####################################
## Phasespace 
#find_library(PHASESPACE_LIBRARIES
#  NAMES
#  owlsock
#  PATHS
#  "${CMAKE_CURRENT_SOURCE_DIR}/phasespace/lib"
#  )
#
####################################
## LABJACK u6 code and library 
#file(GLOB LJ_SOURCES "labjack/*.c" )
#find_library(LABJACK_LIBRARIES
#  NAMES
#  labjackusb
#  PATHS
#  "/usr/local/lib"
#  )



###################################
# ATI NANO 25 Sensors
file(GLOB OTHER_SOURCES "atidaq/*.c" "labjack/*.c")
list(REMOVE_ITEM OTHER_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/atidaq/xmltok_impl.c
  ${CMAKE_CURRENT_SOURCE_DIR}/atidaq/xmltok_ns.c
  ${CMAKE_CURRENT_SOURCE_DIR}/atidaq/events.c )

###################################
include_directories(
  "./"
  "./atidaq/"
  "./labjack/"
  "./phasespace/include/"
  )
link_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/phasespace/lib"
)

set(DARWIN_HW_LIBS phidget21 owlsock labjackusb)

  #${PHIDGETS_LIBRARIES}
  #${PHASESPACE_LIBRARIES}
  #${LABJACK_LIBRARIES})

message(STATUS ${DARWIN_HW_LIBS})
message(STATUS ${OTHER_SOURCES})

# parallel mujoco option from top-level cmake
#if(MUJOCO_SIMULATOR_PARALLEL)
#  find_package(OpenMP REQUIRED)
#  if (OPENMP_FOUND)
#	add_definitions(-DMUJOCO_SIMULATOR_PARALLEL)
#	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#  endif()
#endif(MUJOCO_SIMULATOR_PARALLEL)

#add_library(darwin_hw SHARED ${DARWIN_SOURCES} ${OTHER_SOURCES} ${UJ_SOURCES})
add_library(darwin_hw ${DARWIN_SOURCES} ${OTHER_SOURCES})
target_link_libraries(darwin_hw ${DARWIN_HW_LIBS})

add_executable(test_darwin "test_darwin.cpp" )
target_link_libraries(test_darwin darwin_hw )

#target_link_libraries(test_darwin darwin_hw ${PHIDGETS_LIBRARIES} ${PHASESPACE_LIBRARIES})
#target_include_directories(viewer PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

#add_executable(test_contacts "test_contacts.cpp")
#target_link_libraries(test_contacts ${DARWIN_HW_LIBS} )
#target_include_directories(viewer PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

