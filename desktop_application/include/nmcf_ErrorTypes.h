#pragma once

/**
 * @author David Muttenthaler
 * @date 25-06-2025
 *
 * @brief Error Types
 *
 */

enum class ErrorType {
    ok = 0, // No error
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
