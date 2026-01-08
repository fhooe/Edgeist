/**
 * @file
 * @author David Muttenthaler
 * @brief Implements the NMCF error types
 */

#ifndef EDGEIST_NMCF_ERROR_TYPES_H
#define EDGEIST_NMCF_ERROR_TYPES_H

namespace Edgeist {
/**
 * @brief NMCF error types
 */
enum class ErrorType {
    OK = 0, // No error
    OutOfMemory, // Memory allocation failure
    FileNotFound, // File could not be found
    ModelNotInitialized, // Model not initialized
    LayerNotInitialized, // Model not initialized
    UnknownError, // Unspecified error
    IndexOutOfBounds, // Index out of bounds
    InvalidPointer,
    MissingCachedInputs,
    DropoutMaskMissing
};
} // namespace Edgeist

#endif // EDGEIST_NMCF_ERROR_TYPES_H
