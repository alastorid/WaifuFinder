#ifndef TRUFOR_DIRECT_ANE_H
#define TRUFOR_DIRECT_ANE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <IOSurface/IOSurface.h>

typedef struct TruForDirectANEKernel TruForDirectANEKernel;

/// Experimental, process-isolated bridge. It resolves undocumented Apple APIs
/// dynamically and does not link private framework symbols into the app.
bool trufor_direct_ane_available(char *error, size_t error_capacity);
TruForDirectANEKernel *trufor_direct_ane_compile(
    const char *mil_text, size_t mil_length,
    int weight_count, const char *const *weight_names,
    const uint8_t *const *weight_data, const size_t *weight_lengths,
    size_t input_bytes, size_t output_bytes,
    char *error, size_t error_capacity);
void *trufor_direct_ane_lock_input(TruForDirectANEKernel *kernel);
void trufor_direct_ane_unlock_input(TruForDirectANEKernel *kernel);
const void *trufor_direct_ane_lock_output(TruForDirectANEKernel *kernel);
void trufor_direct_ane_unlock_output(TruForDirectANEKernel *kernel);
bool trufor_direct_ane_evaluate(TruForDirectANEKernel *kernel, char *error, size_t error_capacity);
int trufor_direct_ane_compile_count(void);
void trufor_direct_ane_free(TruForDirectANEKernel *kernel);

#endif
