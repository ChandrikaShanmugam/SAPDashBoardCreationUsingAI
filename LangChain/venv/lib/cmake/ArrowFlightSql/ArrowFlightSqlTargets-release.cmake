#----------------------------------------------------------------
# Generated CMake target import file for configuration "RELEASE".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ArrowFlightSql::arrow_flight_sql_shared" for configuration "RELEASE"
set_property(TARGET ArrowFlightSql::arrow_flight_sql_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ArrowFlightSql::arrow_flight_sql_shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "/Users/chandrika/Documents/LangChain/venv/lib/libarrow_flight_sql.1900.0.0.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libarrow_flight_sql.1900.dylib"
  )

list(APPEND _cmake_import_check_targets ArrowFlightSql::arrow_flight_sql_shared )
list(APPEND _cmake_import_check_files_for_ArrowFlightSql::arrow_flight_sql_shared "/Users/chandrika/Documents/LangChain/venv/lib/libarrow_flight_sql.1900.0.0.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
