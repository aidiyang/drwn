# protobuf compilation
#file(GLOB ProtoFiles "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")
#PROTOBUF_GENERATE_CPP(ProtoSources ProtoHeaders ${ProtoFiles})

file(GLOB DARWIN_SOURCES "*.cpp" )
list(REMOVE_ITEM DARWIN_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/test_contacts.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_darwin.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/calibrate_ps.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/loop_time.cpp 
  ${CMAKE_CURRENT_SOURCE_DIR}/test_eigen.cpp )
#${CMAKE_CURRENT_SOURCE_DIR}/walker.cpp )
if (WIN32)
  list(REMOVE_ITEM DARWIN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/LinuxCM730.cpp )
else ()
  list(REMOVE_ITEM DARWIN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/WindowsCM730.cpp )
endif (WIN32)

message(STATUS ${DARWIN_SOURCES})

find_package(Boost 1.55 COMPONENTS program_options random)

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
#include_directories(
#  "./"
#  "./atidaq/"
#  "./labjack/"
#  "./phasespace/include/"
#  )
link_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/phasespace/lib"
)

set(DARWIN_HW_LIBS phidget21 owlsock labjackusb)
set(DARWIN_INCLUDES  
  "${CMAKE_CURRENT_SOURCE_DIR}/"
  "${CMAKE_CURRENT_SOURCE_DIR}/atidaq/"
  "${CMAKE_CURRENT_SOURCE_DIR}/labjack/"
  "${CMAKE_CURRENT_SOURCE_DIR}/phasespace/include/"
  )

include_directories(
  ${DARWIN_INCLUDES}
  )

message(STATUS ${DARWIN_HW_LIBS})
message(STATUS ${OTHER_SOURCES})
message(STATUS ${DARWIN_INCLUDES})

#add_library(darwin_hw SHARED ${DARWIN_SOURCES} ${OTHER_SOURCES} ${UJ_SOURCES})
add_library(darwin_hw SHARED ${DARWIN_SOURCES} ${OTHER_SOURCES})
target_include_directories(darwin_hw PUBLIC ${DARWIN_INCLUDES} )
target_link_libraries(darwin_hw ${DARWIN_HW_LIBS})

add_executable(test_darwin "test_darwin.cpp" )
target_link_libraries(test_darwin darwin_hw ${Boost_LIBRARIES})

add_executable(dxl_monitor "dxl_monitor.cpp" "dxl_monitor_process.cpp" )
target_link_libraries(dxl_monitor darwin_hw )

add_executable(calibrate_ps "calibrate_ps.cpp" )
target_link_libraries(calibrate_ps darwin_hw )

#add_executable(test_eigen "test_eigen.cpp" )

#target_link_libraries(test_darwin darwin_hw ${PHIDGETS_LIBRARIES} ${PHASESPACE_LIBRARIES})
#target_include_directories(viewer PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

#add_executable(test_contacts "test_contacts.cpp")
#target_link_libraries(test_contacts ${DARWIN_HW_LIBS} )
#target_include_directories(viewer PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

