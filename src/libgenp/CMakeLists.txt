# Create a library with the listed source files.
add_library( genp STATIC
	gdats/PM2DVectorFields.cpp
	gdats/PMBatchStrData.cpp
	gdats/PMBatchStrDataIndex.cpp
	gdats/InputFilelist.cpp
	gdats/FlexDataRead.cpp
	gdats/TdDataReader.cpp
#
	goutp/TdClustWriter.cpp
	goutp/TdAlnWriter.cpp
	goutp/TdAlnWriterPlain.cpp
	goutp/TdAlnWriterJSON.cpp
	goutp/TdFinalizer.cpp
	goutp/TdFinalizerPlainComplex.cpp
	goutp/TdFinalizerJSONComplex.cpp
)

if(GPUINUSE)
  set_source_files_properties(goutp/TdFinalizer.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(goutp/TdFinalizerPlainComplex.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(goutp/TdFinalizerJSONComplex.cpp PROPERTIES LANGUAGE CUDA)
endif()

# Make sure the compiler can find include files for the library
# when other libraries or executables link to it
target_include_directories( genp PUBLIC 
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
##        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

##set_target_properties( genp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link to the required libraries 
##target_link_libraries( mycu PUBLIC pro )

