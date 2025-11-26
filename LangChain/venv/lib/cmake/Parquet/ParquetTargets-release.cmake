#----------------------------------------------------------------
# Generated CMake target import file for configuration "RELEASE".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Parquet::parquet_shared" for configuration "RELEASE"
set_property(TARGET Parquet::parquet_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Parquet::parquet_shared PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "thrift::thrift"
  IMPORTED_LOCATION_RELEASE "/Users/chandrika/Documents/LangChain/venv/lib/libparquet.1900.0.0.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libparquet.1900.dylib"
  )

list(APPEND _cmake_import_check_targets Parquet::parquet_shared )
list(APPEND _cmake_import_check_files_for_Parquet::parquet_shared "/Users/chandrika/Documents/LangChain/venv/lib/libparquet.1900.0.0.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
