# Add executable that is built from the listed source files.
# The extensions are automatically found.
add_executable( gtcomplex 
	gtcomplex.cpp 
)


file(TO_NATIVE_PATH "${CMAKE_INSTALL_PREFIX}/${LOCALSTATEDIR}" GTCOMPLEXSTATEDIRNAT)
string(REGEX REPLACE "([\\])" "\\\\\\1" GTCOMPLEXSTATEDIRNAT "${GTCOMPLEXSTATEDIRNAT}")


## ONE executable
## Specify include directories
target_include_directories( gtcomplex PUBLIC 
	${CMAKE_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions( gtcomplex PUBLIC
	GTCOMPLEX_INSTALL_DIR=${CMAKE_INSTALL_PREFIX}
	GTCOMPLEXSTATEDIR=${GTCOMPLEXSTATEDIRNAT}
)

# Link the executable to the libraries 
if(GPUINUSE)
  set_target_properties( gtcomplex PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties( mycu PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries( gtcomplex PUBLIC mycu mympbase genp util extz Threads::Threads)
else()
  target_link_libraries( gtcomplex PUBLIC mymp mympbase genp util extz Threads::Threads)
endif()


## install targets
install(TARGETS 
	gtcomplex 
	DESTINATION ${LOCALBINDIR}
)
#install(FILES 
#	gtcomplex.sh 
#	DESTINATION ${LOCALBINDIR} 
#	PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_READ WORLD_READ
#)

