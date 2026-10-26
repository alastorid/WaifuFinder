#import "WDTaggerANE.h"
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <dispatch/dispatch.h>
#import <string.h>
#import <CommonCrypto/CommonDigest.h>

static NSString *const WDANEErrorDomain = @"WDTaggerANE";
static Class WDANEDescriptorClass;
static Class WDANEModelClass;
static Class WDANERequestClass;
static Class WDANESurfaceObjectClass;
static BOOL WDANEReady;
static NSError *WDANEInitializationError;
static dispatch_once_t WDANEInitializationOnce;
static void *WDANEFrameworkHandle;
static NSMutableDictionary<NSString *, id> *WDANEProgramCache;
static NSMutableArray<NSString *> *WDANEProgramCacheLRU;
static dispatch_semaphore_t WDANEProgramCacheLock;
static dispatch_once_t WDANEProgramCacheOnce;
static const NSUInteger WDANEProgramCacheCapacity = 2;
static uint64_t WDANEProgramCacheHits;

@interface WDANEProgramBox : NSObject
@property(nonatomic, strong) id model;
@property(nonatomic, copy) NSString *temporaryDirectory;
@property(nonatomic) dispatch_semaphore_t evaluationLock;
@property(nonatomic) BOOL loaded;
@end

@implementation WDANEProgramBox
- (void)dealloc {
    if (self.loaded && self.model) {
        NSError *error = nil;
        SEL selector = NSSelectorFromString(@"unloadWithQoS:error:");
        ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            self.model, selector, 21, &error);
    }
    if (self.temporaryDirectory.length > 0) {
        [[NSFileManager defaultManager] removeItemAtPath:self.temporaryDirectory error:nil];
    }
}
@end

static void WDANEInitializeProgramCache(void) {
    dispatch_once(&WDANEProgramCacheOnce, ^{
        WDANEProgramCache = [NSMutableDictionary dictionary];
        WDANEProgramCacheLRU = [NSMutableArray array];
        WDANEProgramCacheLock = dispatch_semaphore_create(1);
    });
}

static NSString *WDANEProgramCacheKey(NSData *milData, NSDictionary *weightDictionary) {
    if (milData.length > UINT32_MAX) return nil;
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    uint64_t length = (uint64_t)milData.length;
    CC_SHA256_Update(&context, (const uint8_t *)&length, (CC_LONG)sizeof(length));
    CC_SHA256_Update(&context, milData.bytes, (CC_LONG)milData.length);
    NSArray<NSString *> *paths = [[weightDictionary allKeys] sortedArrayUsingSelector:@selector(compare:)];
    for (NSString *path in paths) {
        NSData *pathData = [path dataUsingEncoding:NSUTF8StringEncoding];
        NSDictionary *weightEntry = [weightDictionary[path] isKindOfClass:[NSDictionary class]]
            ? weightDictionary[path] : nil;
        NSData *blobData = weightEntry[@"data"];
        if (!pathData || pathData.length > UINT32_MAX ||
            ![blobData isKindOfClass:[NSData class]] || blobData.length > UINT32_MAX ||
            ![weightEntry[@"offset"] isKindOfClass:[NSNumber class]]) return nil;
        uint64_t pathLength = (uint64_t)pathData.length;
        uint64_t blobLength = (uint64_t)blobData.length;
        uint64_t blobOffset = [weightEntry[@"offset"] unsignedLongLongValue];
        CC_SHA256_Update(&context, (const uint8_t *)&pathLength, (CC_LONG)sizeof(pathLength));
        CC_SHA256_Update(&context, pathData.bytes, (CC_LONG)pathData.length);
        CC_SHA256_Update(&context, (const uint8_t *)&blobOffset, (CC_LONG)sizeof(blobOffset));
        CC_SHA256_Update(&context, (const uint8_t *)&blobLength, (CC_LONG)sizeof(blobLength));
        CC_SHA256_Update(&context, blobData.bytes, (CC_LONG)blobData.length);
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);
    NSMutableString *key = [NSMutableString stringWithCapacity:CC_SHA256_DIGEST_LENGTH * 2];
    for (NSUInteger i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) [key appendFormat:@"%02x", digest[i]];
    return key;
}

static WDANEProgram *WDANEProgramCacheLookup(NSString *key) {
    WDANEInitializeProgramCache();
    dispatch_semaphore_wait(WDANEProgramCacheLock, DISPATCH_TIME_FOREVER);
    id box = WDANEProgramCache[key];
    if (box) {
        WDANEProgramCacheHits++;
        [WDANEProgramCacheLRU removeObject:key];
        [WDANEProgramCacheLRU addObject:key];
    }
    WDANEProgram *program = box ? (WDANEProgram *)CFBridgingRetain(box) : NULL;
    dispatch_semaphore_signal(WDANEProgramCacheLock);
    return program;
}

static WDANEProgram *WDANEProgramCacheInsert(NSString *key, id newBox, WDANEProgram *newProgram,
                                             NSString *temporaryDirectory) {
    WDANEInitializeProgramCache();
    dispatch_semaphore_wait(WDANEProgramCacheLock, DISPATCH_TIME_FOREVER);
    id existingBox = WDANEProgramCache[key];
    if (existingBox) {
        [WDANEProgramCacheLRU removeObject:key];
        [WDANEProgramCacheLRU addObject:key];
        dispatch_semaphore_signal(WDANEProgramCacheLock);
        CFRelease((CFTypeRef)newProgram);
        [[NSFileManager defaultManager] removeItemAtPath:temporaryDirectory error:nil];
        return (WDANEProgram *)CFBridgingRetain(existingBox);
    }
    WDANEProgramCache[key] = newBox;
    [WDANEProgramCacheLRU addObject:key];
    while (WDANEProgramCacheLRU.count > WDANEProgramCacheCapacity) {
        NSString *oldest = WDANEProgramCacheLRU.firstObject;
        [WDANEProgramCacheLRU removeObjectAtIndex:0];
        [WDANEProgramCache removeObjectForKey:oldest];
    }
    dispatch_semaphore_signal(WDANEProgramCacheLock);
    return newProgram;
}

@interface WDANETensorBox : NSObject
@property(nonatomic) IOSurfaceRef surface;
@property(nonatomic) size_t elementCount;
@property(nonatomic) size_t allocationBytes;
@end

@implementation WDANETensorBox
- (void)dealloc {
    if (self.surface) CFRelease(self.surface);
}
@end

static void WDANEInitialize(void) {
    dispatch_once(&WDANEInitializationOnce, ^{
        NSString *frameworkPath = @"/System/Library/PrivateFrameworks/"
                                  @"AppleNeuralEngine.framework/AppleNeuralEngine";
        WDANEFrameworkHandle = dlopen(frameworkPath.fileSystemRepresentation, RTLD_NOW | RTLD_LOCAL);
        if (!WDANEFrameworkHandle) {
            WDANEInitializationError = [NSError errorWithDomain:WDANEErrorDomain code:1
                userInfo:@{NSLocalizedDescriptionKey: @"AppleNeuralEngine private framework is unavailable."}];
            return;
        }
        WDANEDescriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        WDANEModelClass = NSClassFromString(@"_ANEInMemoryModel");
        WDANERequestClass = NSClassFromString(@"_ANERequest");
        WDANESurfaceObjectClass = NSClassFromString(@"_ANEIOSurfaceObject");
        if (!WDANEDescriptorClass || !WDANEModelClass || !WDANERequestClass || !WDANESurfaceObjectClass) {
            WDANEInitializationError = [NSError errorWithDomain:WDANEErrorDomain code:2
                userInfo:@{NSLocalizedDescriptionKey: @"Required private ANE runtime classes were not found."}];
            return;
        }
        WDANEReady = YES;
    });
}

static void WDANEWriteError(char *destination, size_t capacity, NSError *error, NSString *message) {
    if (!destination || capacity == 0) return;
    NSString *text = error.localizedDescription ?: message;
    strlcpy(destination, text.UTF8String ?: "Unknown direct ANE error.", capacity);
}

BOOL WDANEIsAvailable(void) {
    WDANEInitialize();
    return WDANEReady;
}

static WDANEProgram *WDANECompileMILInternal(const char *milText, NSDictionary *weightDictionary,
                                             char *errorMessage, size_t errorCapacity) {
    WDANEInitialize();
    if (!WDANEReady) {
        WDANEWriteError(errorMessage, errorCapacity, WDANEInitializationError, @"Direct ANE runtime is unavailable.");
        return NULL;
    }
    if (!milText || milText[0] == '\0') {
        WDANEWriteError(errorMessage, errorCapacity, nil, @"MIL source is empty.");
        return NULL;
    }

    @autoreleasepool {
        NSData *milData = [NSData dataWithBytes:milText length:strlen(milText)];
        NSString *cacheKey = WDANEProgramCacheKey(milData, weightDictionary ?: @{});
        if (cacheKey) {
            WDANEProgram *cachedProgram = WDANEProgramCacheLookup(cacheKey);
            if (cachedProgram) return cachedProgram;
        }
        SEL descriptorSelector = NSSelectorFromString(@"modelWithMILText:weights:optionsPlist:");
        id descriptor = ((id (*)(id, SEL, id, id, id))objc_msgSend)(
            WDANEDescriptorClass, descriptorSelector, milData, weightDictionary ?: @{}, nil);
        if (!descriptor) {
            WDANEWriteError(errorMessage, errorCapacity, nil, @"ANE rejected the in-memory MIL descriptor.");
            return NULL;
        }
        SEL modelSelector = NSSelectorFromString(@"inMemoryModelWithDescriptor:");
        id model = ((id (*)(id, SEL, id))objc_msgSend)(WDANEModelClass, modelSelector, descriptor);
        if (!model) {
            WDANEWriteError(errorMessage, errorCapacity, nil, @"ANE could not create an in-memory model.");
            return NULL;
        }

        SEL identifierSelector = NSSelectorFromString(@"hexStringIdentifier");
        NSString *identifier = ((id (*)(id, SEL))objc_msgSend)(model, identifierSelector);
        if (identifier.length == 0) {
            WDANEWriteError(errorMessage, errorCapacity, nil, @"ANE did not provide a model identifier.");
            return NULL;
        }
        NSString *directory = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
        NSFileManager *fileManager = [NSFileManager defaultManager];
        NSError *fileError = nil;
        [fileManager createDirectoryAtPath:[directory stringByAppendingPathComponent:@"weights"]
                withIntermediateDirectories:YES attributes:nil error:&fileError];
        if (fileError || ![milData writeToFile:[directory stringByAppendingPathComponent:@"model.mil"]
                                       options:NSDataWritingAtomic error:&fileError]) {
            WDANEWriteError(errorMessage, errorCapacity, fileError, @"Could not prepare the fixed-shape MIL program.");
            [fileManager removeItemAtPath:directory error:nil];
            return NULL;
        }
        for (NSString *blobPath in weightDictionary) {
            if (![blobPath hasPrefix:@"@model_path/weights/"] ||
                [blobPath containsString:@".."] || ![weightDictionary[blobPath] isKindOfClass:[NSDictionary class]]) {
                WDANEWriteError(errorMessage, errorCapacity, nil, @"ANE weight blobs must use safe @model_path/weights/ paths.");
                [fileManager removeItemAtPath:directory error:nil];
                return NULL;
            }
            NSData *blobData = weightDictionary[blobPath][@"data"];
            if (![blobData isKindOfClass:[NSData class]] || blobData.length < 128) {
                WDANEWriteError(errorMessage, errorCapacity, nil, @"ANE weight blob is missing a complete BLOBFILE payload.");
                [fileManager removeItemAtPath:directory error:nil];
                return NULL;
            }
            NSString *relativePath = [blobPath stringByReplacingOccurrencesOfString:@"@model_path/"
                                                                         withString:@""];
            NSString *fullPath = [directory stringByAppendingPathComponent:relativePath];
            [fileManager createDirectoryAtPath:[fullPath stringByDeletingLastPathComponent]
                    withIntermediateDirectories:YES attributes:nil error:&fileError];
            if (fileError || ![blobData writeToFile:fullPath options:NSDataWritingAtomic error:&fileError]) {
                WDANEWriteError(errorMessage, errorCapacity, fileError, @"Could not write an ANE BLOBFILE weight.");
                [fileManager removeItemAtPath:directory error:nil];
                return NULL;
            }
        }

        SEL compileSelector = NSSelectorFromString(@"compileWithQoS:options:error:");
        BOOL compiled = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, compileSelector, 21, @{}, &fileError);
        if (!compiled) {
            WDANEWriteError(errorMessage, errorCapacity, fileError, @"The private ANE compiler rejected the MIL program.");
            [fileManager removeItemAtPath:directory error:nil];
            return NULL;
        }
        SEL loadSelector = NSSelectorFromString(@"loadWithQoS:options:error:");
        BOOL loaded = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, loadSelector, 21, @{}, &fileError);
        if (!loaded) {
            WDANEWriteError(errorMessage, errorCapacity, fileError, @"The compiled ANE program could not be loaded.");
            [fileManager removeItemAtPath:directory error:nil];
            return NULL;
        }

        WDANEProgramBox *box = [WDANEProgramBox new];
        box.model = model;
        box.temporaryDirectory = directory;
        box.evaluationLock = dispatch_semaphore_create(1);
        box.loaded = YES;
        WDANEProgram *newProgram = (WDANEProgram *)CFBridgingRetain(box);
        return cacheKey
            ? WDANEProgramCacheInsert(cacheKey, box, newProgram, directory)
            : newProgram;
    }
}

WDANEProgram *WDANECompileMIL(const char *milText, char *errorMessage, size_t errorCapacity) {
    return WDANECompileMILInternal(milText, @{}, errorMessage, errorCapacity);
}

WDANEProgram *WDANECompileMILWithWeightBlobs(const char *milText,
                                             const char *const *paths,
                                             const uint8_t *const *blobs,
                                             const size_t *blobSizes,
                                             size_t blobCount,
                                             char *errorMessage,
                                             size_t errorCapacity) {
    if (blobCount > 0 && (!paths || !blobs || !blobSizes)) {
        WDANEWriteError(errorMessage, errorCapacity, nil, @"ANE weight blob arrays are incomplete.");
        return NULL;
    }
    NSMutableDictionary *dictionary = [NSMutableDictionary dictionaryWithCapacity:blobCount];
    for (size_t index = 0; index < blobCount; index++) {
        if (!paths[index] || !blobs[index] || blobSizes[index] < 128) {
            WDANEWriteError(errorMessage, errorCapacity, nil, @"ANE weight blob path or data is invalid.");
            return NULL;
        }
        NSString *path = [NSString stringWithUTF8String:paths[index]];
        NSData *data = [NSData dataWithBytes:blobs[index] length:blobSizes[index]];
        dictionary[path] = @{@"offset": @0, @"data": data};
    }
    return WDANECompileMILInternal(milText, dictionary, errorMessage, errorCapacity);
}

WDANEProgram *WDANECompileMILWithTwoWeightBlobs(const char *milText,
                                                 const char *weightPath,
                                                 const uint8_t *weightBlob,
                                                 size_t weightBlobSize,
                                                 const char *biasPath,
                                                 const uint8_t *biasBlob,
                                                 size_t biasBlobSize,
                                                 char *errorMessage,
                                                 size_t errorCapacity) {
    const char *paths[] = {weightPath, biasPath};
    const uint8_t *blobs[] = {weightBlob, biasBlob};
    const size_t sizes[] = {weightBlobSize, biasBlobSize};
    return WDANECompileMILWithWeightBlobs(milText, paths, blobs, sizes, 2,
                                          errorMessage, errorCapacity);
}

WDANEProgram *WDANECompileMILWithOneWeightBlob(const char *milText,
                                                const char *weightPath,
                                                const uint8_t *weightBlob,
                                                size_t weightBlobSize,
                                                char *errorMessage,
                                                size_t errorCapacity) {
    const char *paths[] = {weightPath};
    const uint8_t *blobs[] = {weightBlob};
    const size_t sizes[] = {weightBlobSize};
    return WDANECompileMILWithWeightBlobs(milText, paths, blobs, sizes, 1,
                                          errorMessage, errorCapacity);
}

void WDANEReleaseProgram(WDANEProgram *program) {
    if (program) CFRelease((CFTypeRef)program);
}

uint64_t WDANEProgramCacheHitCount(void) {
    WDANEInitializeProgramCache();
    dispatch_semaphore_wait(WDANEProgramCacheLock, DISPATCH_TIME_FOREVER);
    uint64_t hits = WDANEProgramCacheHits;
    dispatch_semaphore_signal(WDANEProgramCacheLock);
    return hits;
}

size_t WDANEProgramCacheEntryCount(void) {
    WDANEInitializeProgramCache();
    dispatch_semaphore_wait(WDANEProgramCacheLock, DISPATCH_TIME_FOREVER);
    size_t count = WDANEProgramCache.count;
    dispatch_semaphore_signal(WDANEProgramCacheLock);
    return count;
}

WDANETensor *WDANETensorCreateF16(size_t elementCount, size_t minimumAllocationBytes,
                                   char *errorMessage, size_t errorCapacity) {
    if (elementCount == 0 || elementCount > SIZE_MAX / sizeof(uint16_t)) {
        WDANEWriteError(errorMessage, errorCapacity, nil, @"Invalid FP16 tensor size.");
        return NULL;
    }
    size_t dataBytes = elementCount * sizeof(uint16_t);
    size_t allocationBytes = MAX(dataBytes, minimumAllocationBytes);
    if (allocationBytes > UINT32_MAX) {
        WDANEWriteError(errorMessage, errorCapacity, nil, @"IOSurface allocation exceeds the supported size.");
        return NULL;
    }
    NSDictionary *properties = @{
        (id)kIOSurfaceWidth: @(allocationBytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(allocationBytes),
        (id)kIOSurfaceAllocSize: @(allocationBytes),
        (id)kIOSurfacePixelFormat: @0,
    };
    IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)properties);
    if (!surface) {
        WDANEWriteError(errorMessage, errorCapacity, nil, @"IOSurface allocation failed.");
        return NULL;
    }
    WDANETensorBox *box = [WDANETensorBox new];
    box.surface = surface;
    box.elementCount = elementCount;
    box.allocationBytes = IOSurfaceGetAllocSize(surface);
    return (WDANETensor *)CFBridgingRetain(box);
}

void WDANETensorRelease(WDANETensor *tensor) {
    if (tensor) CFRelease((CFTypeRef)tensor);
}

size_t WDANETensorElementCount(WDANETensor *tensor) {
    return tensor ? ((__bridge WDANETensorBox *)tensor).elementCount : 0;
}

size_t WDANETensorAllocationBytes(WDANETensor *tensor) {
    return tensor ? ((__bridge WDANETensorBox *)tensor).allocationBytes : 0;
}

static BOOL WDANETensorCopy(WDANETensor *tensor, void *values, size_t count,
                            BOOL write, char *errorMessage, size_t errorCapacity) {
    if (!tensor || !values || count != WDANETensorElementCount(tensor)) {
        WDANEWriteError(errorMessage, errorCapacity, nil, @"FP16 tensor element count does not match its fixed shape.");
        return NO;
    }
    WDANETensorBox *box = (__bridge WDANETensorBox *)tensor;
    size_t byteCount = count * sizeof(uint16_t);
    IOReturn lockResult = IOSurfaceLock(box.surface, write ? 0 : kIOSurfaceLockReadOnly, NULL);
    if (lockResult != kIOReturnSuccess) {
        WDANEWriteError(errorMessage, errorCapacity, nil, @"Could not lock the IOSurface tensor.");
        return NO;
    }
    void *base = IOSurfaceGetBaseAddress(box.surface);
    BOOL valid = base != NULL && IOSurfaceGetAllocSize(box.surface) >= byteCount;
    if (valid) {
        if (write) memcpy(base, values, byteCount);
        else memcpy(values, base, byteCount);
    }
    IOSurfaceUnlock(box.surface, write ? 0 : kIOSurfaceLockReadOnly, NULL);
    if (!valid) WDANEWriteError(errorMessage, errorCapacity, nil, @"IOSurface storage is smaller than the fixed tensor shape.");
    return valid;
}

BOOL WDANETensorWriteF16(WDANETensor *tensor, const uint16_t *values, size_t count,
                         char *errorMessage, size_t errorCapacity) {
    return WDANETensorCopy(tensor, (void *)values, count, YES, errorMessage, errorCapacity);
}

BOOL WDANETensorReadF16(WDANETensor *tensor, uint16_t *values, size_t count,
                        char *errorMessage, size_t errorCapacity) {
    return WDANETensorCopy(tensor, values, count, NO, errorMessage, errorCapacity);
}

BOOL WDANEEvaluate(WDANEProgram *program, WDANETensor *const *inputs, size_t inputCount,
                   WDANETensor *const *outputs, size_t outputCount,
                   char *errorMessage, size_t errorCapacity) {
    if (!WDANEReady || !program || !inputs || !outputs || inputCount == 0 || outputCount == 0) {
        WDANEWriteError(errorMessage, errorCapacity, nil, @"Invalid direct ANE evaluation request.");
        return NO;
    }
    @autoreleasepool {
        NSMutableArray *inputObjects = [NSMutableArray arrayWithCapacity:inputCount];
        NSMutableArray *inputIndices = [NSMutableArray arrayWithCapacity:inputCount];
        NSMutableArray *outputObjects = [NSMutableArray arrayWithCapacity:outputCount];
        NSMutableArray *outputIndices = [NSMutableArray arrayWithCapacity:outputCount];
        SEL surfaceSelector = NSSelectorFromString(@"objectWithIOSurface:");
        id (*surfaceFactory)(id, SEL, IOSurfaceRef) = (void *)objc_msgSend;
        for (size_t i = 0; i < inputCount; i++) {
            if (!inputs[i]) {
                WDANEWriteError(errorMessage, errorCapacity, nil, @"Direct ANE input tensor is missing.");
                return NO;
            }
            WDANETensorBox *tensor = (__bridge WDANETensorBox *)inputs[i];
            id wrapped = surfaceFactory(WDANESurfaceObjectClass, surfaceSelector, tensor.surface);
            if (!wrapped) {
                WDANEWriteError(errorMessage, errorCapacity, nil, @"ANE could not wrap an input IOSurface.");
                return NO;
            }
            [inputObjects addObject:wrapped];
            [inputIndices addObject:@(i)];
        }
        for (size_t i = 0; i < outputCount; i++) {
            if (!outputs[i]) {
                WDANEWriteError(errorMessage, errorCapacity, nil, @"Direct ANE output tensor is missing.");
                return NO;
            }
            WDANETensorBox *tensor = (__bridge WDANETensorBox *)outputs[i];
            id wrapped = surfaceFactory(WDANESurfaceObjectClass, surfaceSelector, tensor.surface);
            if (!wrapped) {
                WDANEWriteError(errorMessage, errorCapacity, nil, @"ANE could not wrap an output IOSurface.");
                return NO;
            }
            [outputObjects addObject:wrapped];
            [outputIndices addObject:@(i)];
        }

        WDANEProgramBox *box = (__bridge WDANEProgramBox *)program;
        SEL requestSelector = NSSelectorFromString(
            @"requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:");
        id request = ((id (*)(id, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            WDANERequestClass, requestSelector, inputObjects, inputIndices,
            outputObjects, outputIndices, nil, nil, @0);
        if (!request) {
            WDANEWriteError(errorMessage, errorCapacity, nil, @"ANE could not create an evaluation request.");
            return NO;
        }
        NSError *evaluationError = nil;
        SEL evaluateSelector = NSSelectorFromString(@"evaluateWithQoS:options:request:error:");
        dispatch_semaphore_t evaluationLock = box.evaluationLock;
        dispatch_semaphore_wait(evaluationLock, DISPATCH_TIME_FOREVER);
        BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            box.model, evaluateSelector, 21, @{}, request, &evaluationError);
        dispatch_semaphore_signal(evaluationLock);
        if (!ok) WDANEWriteError(errorMessage, errorCapacity, evaluationError, @"Direct ANE evaluation failed.");
        return ok;
    }
}
