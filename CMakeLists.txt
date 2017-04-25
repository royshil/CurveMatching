cmake_minimum_required (VERSION 3.2) 
project (CurveMatching) 

set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

include(cotire)

find_package(OpenCV REQUIRED)

file(GLOB MYSOURCES	*.cpp *.c)
file(GLOB MYHEADERS	*.h *.hpp)

add_executable(MatchTwoCurves 
	${MYSOURCES} ${MYHEADERS}
	)
	
target_link_libraries(MatchTwoCurves
	${OpenCV_LIBS}
	)
	
set_target_properties(MatchTwoCurves PROPERTIES COTIRE_CXX_PREFIX_HEADER_INIT "std.h")

cotire(MatchTwoCurves)
