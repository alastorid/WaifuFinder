#pragma once

#import <Foundation/Foundation.h>
#import <stddef.h>
#import <stdint.h>

NS_ASSUME_NONNULL_BEGIN

typedef struct WDANEProgram WDANEProgram;
typedef struct WDANETensor WDANETensor;

/// Loads Apple's private ANE framework and checks the runtime classes used here.
/// This function never starts a Core ML or Metal fallback.
FOUNDATION_EXPORT BOOL WDANEIsAvailable(void);

/// Compiles and loads a fixed-shape, weight-free MIL program.
FOUNDATION_EXPORT WDANEProgram * _Nullable WDANECompileMIL(const char *milText,
                                                           char * _Nullable errorMessage,
                                                           size_t errorCapacity);
/// Compiles a fixed-shape MIL program with immutable BLOBFILE weights.
/// Each path must be under @model_path/weights/ and each blob contains its
/// 128-byte BLOBFILE header followed by the tensor bytes.
FOUNDATION_EXPORT WDANEProgram * _Nullable WDANECompileMILWithWeightBlobs(
    const char *milText, const char * _Nonnull const * _Nonnull paths,
    const uint8_t * _Nonnull const * _Nonnull blobs, const size_t * _Nonnull blobSizes,
    size_t blobCount, char * _Nullable errorMessage, size_t errorCapacity);
FOUNDATION_EXPORT WDANEProgram * _Nullable WDANECompileMILWithTwoWeightBlobs(
    const char *milText, const char *weightPath, const uint8_t *weightBlob, size_t weightBlobSize,
    const char *biasPath, const uint8_t *biasBlob, size_t biasBlobSize,
    char * _Nullable errorMessage, size_t errorCapacity);
FOUNDATION_EXPORT WDANEProgram * _Nullable WDANECompileMILWithOneWeightBlob(
    const char *milText, const char *weightPath, const uint8_t *weightBlob,
    size_t weightBlobSize, char * _Nullable errorMessage, size_t errorCapacity);
FOUNDATION_EXPORT void WDANEReleaseProgram(WDANEProgram *program);
/// Returns process-local immutable-program cache diagnostics.
FOUNDATION_EXPORT uint64_t WDANEProgramCacheHitCount(void);
FOUNDATION_EXPORT size_t WDANEProgramCacheEntryCount(void);

/// Creates a flat FP16 IOSurface. A caller may request a larger allocation so
/// multiple inputs can use equal-sized surfaces when a program needs that.
FOUNDATION_EXPORT WDANETensor * _Nullable WDANETensorCreateF16(size_t elementCount,
                                                               size_t minimumAllocationBytes,
                                                               char * _Nullable errorMessage,
                                                               size_t errorCapacity);
FOUNDATION_EXPORT void WDANETensorRelease(WDANETensor *tensor);
FOUNDATION_EXPORT size_t WDANETensorElementCount(WDANETensor *tensor);
FOUNDATION_EXPORT size_t WDANETensorAllocationBytes(WDANETensor *tensor);
FOUNDATION_EXPORT BOOL WDANETensorWriteF16(WDANETensor *tensor,
                                           const uint16_t *values,
                                           size_t count,
                                           char * _Nullable errorMessage,
                                           size_t errorCapacity);
FOUNDATION_EXPORT BOOL WDANETensorReadF16(WDANETensor *tensor,
                                          uint16_t *values,
                                          size_t count,
                                          char * _Nullable errorMessage,
                                          size_t errorCapacity);

/// Runs one direct _ANEInMemoryModel evaluation over caller-owned IOSurfaces.
FOUNDATION_EXPORT BOOL WDANEEvaluate(WDANEProgram *program,
                                     WDANETensor * _Nonnull const * _Nonnull inputs,
                                     size_t inputCount,
                                     WDANETensor * _Nonnull const * _Nonnull outputs,
                                     size_t outputCount,
                                     char * _Nullable errorMessage,
                                     size_t errorCapacity);

NS_ASSUME_NONNULL_END
