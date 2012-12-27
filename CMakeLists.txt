cmake_minimum_required (VERSION 2.6) 
project (CurveMatching) 

set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

include(cotire)

find_package(OpenCV 2.4 REQUIRED)

add_executable(MatchTwoCurves 
	match_two_curves.cpp 
	CurveCSS.cpp CurveCSS.h 
	CurveSignature.cpp CurveSignature.h
	std.h
	)
	
target_link_libraries(MatchTwoCurves
	${OpenCV_LIBS}
	mgl
	${OPENCL_LIBRARIES}
	)
	
set_target_properties(MatchTwoCurves PROPERTIES COTIRE_CXX_PREFIX_HEADER_INIT "std.h")
set_target_properties(MatchTwoCurves PROPERTIES COMPILE_FLAGS "-DWITHOUT_OPENCL")

cotire(MatchTwoCurves)