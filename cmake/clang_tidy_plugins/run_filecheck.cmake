execute_process(
  COMMAND
    "${CLANG_TIDY}" "--load=${PLUGIN}" "--config=${CONFIG}"
    "--export-fixes=${OUTPUT}.yaml" -p "${BUILD_DIR}" "${SOURCE}"
  OUTPUT_FILE "${OUTPUT}"
  ERROR_FILE "${OUTPUT}"
  COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
  COMMAND
    "${FILECHECK}" "${SOURCE}" --check-prefix=CHECK-FIXES
    "--input-file=${OUTPUT}.yaml"
  COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
  COMMAND
    "${FILECHECK}" "${SOURCE}" --check-prefix=CHECK-MESSAGES
    "--input-file=${OUTPUT}"
  COMMAND_ERROR_IS_FATAL ANY
)
