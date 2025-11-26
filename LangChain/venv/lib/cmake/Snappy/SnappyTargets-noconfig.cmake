#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Snappy::snappy" for configuration ""
set_property(TARGET Snappy::snappy APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(Snappy::snappy PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsnappy.1.2.1.dylib"
  IMPORTED_SONAME_NOCONFIG "@rpath/libsnappy.1.dylib"
  )

list(APPEND _cmake_import_check_targets Snappy::snappy )
list(APPEND _cmake_import_check_files_for_Snappy::snappy "${_IMPORT_PREFIX}/lib/libsnappy.1.2.1.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
