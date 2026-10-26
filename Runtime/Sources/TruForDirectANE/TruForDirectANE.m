// Experimental direct-ANE bridge adapted from maderix/ANE under MIT.
// This code is used only by a separate research worker, never by the app.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <dlfcn.h>
#import <string.h>
#import "TruForDirectANE.h"

struct TruForDirectANEKernel {
    id model;
    id request;
    NSString *temporaryDirectory;
    IOSurfaceRef input;
    IOSurfaceRef output;
};

static Class g_descriptor;
static Class g_modelClass;
static Class g_requestClass;
static Class g_ioClass;
static bool g_initialized;
static int g_compile_count;

static void set_error(char *output, size_t capacity, NSString *message) {
    if (!output || capacity == 0) return;
    const char *utf8 = message.UTF8String ?: "unknown direct-ANE error";
    strlcpy(output, utf8, capacity);
}

bool trufor_direct_ane_available(char *error, size_t error_capacity) {
    if (g_initialized) return true;
    void *framework = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW | RTLD_GLOBAL);
    if (!framework) {
        set_error(error, error_capacity, [NSString stringWithUTF8String:dlerror() ?: "private ANE framework unavailable"]);
        return false;
    }
    g_descriptor = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_modelClass = NSClassFromString(@"_ANEInMemoryModel");
    g_requestClass = NSClassFromString(@"_ANERequest");
    g_ioClass = NSClassFromString(@"_ANEIOSurfaceObject");
    if (!g_descriptor || !g_modelClass || !g_requestClass || !g_ioClass) {
        set_error(error, error_capacity, @"Private ANE runtime classes are unavailable on this OS build");
        return false;
    }
    g_initialized = true;
    return true;
}

static IOSurfaceRef create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

TruForDirectANEKernel *trufor_direct_ane_compile(
    const char *mil_text, size_t mil_length,
    int weight_count, const char *const *weight_names,
    const uint8_t *const *weight_data, const size_t *weight_lengths,
    size_t input_bytes, size_t output_bytes,
    char *error, size_t error_capacity) {
    @autoreleasepool {
        if (!trufor_direct_ane_available(error, error_capacity)) return NULL;
        if (!mil_text || mil_length == 0 || weight_count < 0 || input_bytes == 0 || output_bytes == 0) {
            set_error(error, error_capacity, @"Invalid direct-ANE compile arguments");
            return NULL;
        }

        NSData *mil = [NSData dataWithBytes:mil_text length:mil_length];
        NSMutableDictionary *weight_dictionary = [NSMutableDictionary dictionaryWithCapacity:(NSUInteger)weight_count];
        for (int i = 0; i < weight_count; ++i) {
            if (!weight_names[i] || !weight_data[i] || weight_lengths[i] == 0) {
                set_error(error, error_capacity, @"Invalid direct-ANE weight blob");
                return NULL;
            }
            NSString *name = [NSString stringWithUTF8String:weight_names[i]];
            NSData *blob = [NSData dataWithBytes:weight_data[i] length:weight_lengths[i]];
            weight_dictionary[name] = @{@"offset": @0, @"data": blob};
        }

        NSError *failure = nil;
        id descriptor = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
            g_descriptor, @selector(modelWithMILText:weights:optionsPlist:),
            mil, weight_dictionary, nil);
        if (!descriptor) {
            set_error(error, error_capacity, @"ANE rejected the MIL model descriptor");
            return NULL;
        }
        id model = ((id (*)(Class, SEL, id))objc_msgSend)(
            g_modelClass, @selector(inMemoryModelWithDescriptor:), descriptor);
        if (!model) {
            set_error(error, error_capacity, @"ANE could not create an in-memory model");
            return NULL;
        }

        NSString *identifier = ((id (*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        if (identifier.length == 0) {
            set_error(error, error_capacity, @"ANE returned an empty model identifier");
            return NULL;
        }
        NSString *temporary_directory = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
        NSFileManager *file_manager = [NSFileManager defaultManager];
        NSString *weights_directory = [temporary_directory stringByAppendingPathComponent:@"weights"];
        if (![file_manager createDirectoryAtPath:weights_directory withIntermediateDirectories:YES attributes:nil error:&failure] ||
            ![mil writeToFile:[temporary_directory stringByAppendingPathComponent:@"model.mil"] atomically:YES]) {
            set_error(error, error_capacity, failure.localizedDescription ?: @"Could not stage direct-ANE model files");
            [file_manager removeItemAtPath:temporary_directory error:nil];
            return NULL;
        }
        for (int i = 0; i < weight_count; ++i) {
            NSString *name = [NSString stringWithUTF8String:weight_names[i]];
            NSString *relative = [name hasPrefix:@"@model_path/"] ? [name substringFromIndex:12] : name;
            NSString *destination = [temporary_directory stringByAppendingPathComponent:relative];
            NSString *parent = [destination stringByDeletingLastPathComponent];
            NSData *blob = weight_dictionary[name][@"data"];
            if (![file_manager createDirectoryAtPath:parent withIntermediateDirectories:YES attributes:nil error:&failure] ||
                ![blob writeToFile:destination atomically:YES]) {
                set_error(error, error_capacity, failure.localizedDescription ?: @"Could not stage direct-ANE weight data");
                [file_manager removeItemAtPath:temporary_directory error:nil];
                return NULL;
            }
        }

        BOOL compiled = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &failure);
        if (!compiled) {
            set_error(error, error_capacity, failure.localizedDescription ?: @"ANE compilation failed");
            [file_manager removeItemAtPath:temporary_directory error:nil];
            return NULL;
        }
        BOOL loaded = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &failure);
        if (!loaded) {
            set_error(error, error_capacity, failure.localizedDescription ?: @"ANE model load failed");
            [file_manager removeItemAtPath:temporary_directory error:nil];
            return NULL;
        }

        IOSurfaceRef input = create_surface(input_bytes);
        IOSurfaceRef output = create_surface(output_bytes);
        if (!input || !output) {
            if (input) CFRelease(input);
            if (output) CFRelease(output);
            ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &failure);
            [file_manager removeItemAtPath:temporary_directory error:nil];
            set_error(error, error_capacity, @"Could not allocate direct-ANE IOSurfaces");
            return NULL;
        }

        id wrapped_input = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            g_ioClass, @selector(objectWithIOSurface:), input);
        id wrapped_output = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            g_ioClass, @selector(objectWithIOSurface:), output);
        id request = ((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            g_requestClass,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wrapped_input], @[@0], @[wrapped_output], @[@0], nil, nil, @0);
        if (!request) {
            CFRelease(input); CFRelease(output);
            ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &failure);
            [file_manager removeItemAtPath:temporary_directory error:nil];
            set_error(error, error_capacity, @"Could not create a direct-ANE request");
            return NULL;
        }

        TruForDirectANEKernel *kernel = calloc(1, sizeof(TruForDirectANEKernel));
        if (!kernel) {
            CFRelease(input); CFRelease(output);
            ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &failure);
            [file_manager removeItemAtPath:temporary_directory error:nil];
            set_error(error, error_capacity, @"Could not allocate direct-ANE kernel handle");
            return NULL;
        }
        kernel->model = model;
        kernel->request = request;
        kernel->temporaryDirectory = temporary_directory;
        kernel->input = input;
        kernel->output = output;
        __sync_fetch_and_add(&g_compile_count, 1);
        return kernel;
    }
}

void *trufor_direct_ane_lock_input(TruForDirectANEKernel *kernel) {
    if (!kernel || !kernel->input || IOSurfaceLock(kernel->input, 0, NULL) != kIOReturnSuccess) return NULL;
    return IOSurfaceGetBaseAddress(kernel->input);
}

void trufor_direct_ane_unlock_input(TruForDirectANEKernel *kernel) {
    if (kernel && kernel->input) IOSurfaceUnlock(kernel->input, 0, NULL);
}

const void *trufor_direct_ane_lock_output(TruForDirectANEKernel *kernel) {
    if (!kernel || !kernel->output || IOSurfaceLock(kernel->output, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return NULL;
    return IOSurfaceGetBaseAddress(kernel->output);
}

void trufor_direct_ane_unlock_output(TruForDirectANEKernel *kernel) {
    if (kernel && kernel->output) IOSurfaceUnlock(kernel->output, kIOSurfaceLockReadOnly, NULL);
}

bool trufor_direct_ane_evaluate(TruForDirectANEKernel *kernel, char *error, size_t error_capacity) {
    @autoreleasepool {
        if (!kernel || !kernel->model || !kernel->request) {
            set_error(error, error_capacity, @"Direct-ANE kernel is not loaded");
            return false;
        }
        NSError *failure = nil;
        BOOL result = ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            kernel->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, kernel->request, &failure);
        if (!result) set_error(error, error_capacity, failure.localizedDescription ?: @"ANE evaluation failed");
        return result;
    }
}

int trufor_direct_ane_compile_count(void) { return g_compile_count; }

void trufor_direct_ane_free(TruForDirectANEKernel *kernel) {
    @autoreleasepool {
        if (!kernel) return;
        NSError *failure = nil;
        if (kernel->model) {
            ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
                kernel->model, @selector(unloadWithQoS:error:), 21, &failure);
        }
        if (kernel->input) CFRelease(kernel->input);
        if (kernel->output) CFRelease(kernel->output);
        if (kernel->temporaryDirectory) {
            [[NSFileManager defaultManager] removeItemAtPath:kernel->temporaryDirectory error:nil];
        }
        kernel->model = nil;
        kernel->request = nil;
        kernel->temporaryDirectory = nil;
        free(kernel);
    }
}
