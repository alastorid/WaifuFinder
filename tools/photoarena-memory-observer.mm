// Final-session observer, injected identically into reference and optimized
// headless clients/workers. No model/image data is inspected or written.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach.h>
#include <pthread.h>
#include <atomic>
#include <unistd.h>
#include <cstdio>
#include <ctime>
static std::atomic<bool> done(false);
static pthread_t thread;
static bool active=false;
static void * sample(void *) {
    @autoreleasepool {
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        const char * directory=getenv("ARENA_MEMORY_DIRECTORY");
        char path[4096]; snprintf(path,sizeof(path),"%s/%d.csv",directory,getpid());
        FILE * file=fopen(path,"w"); if (!file) return nullptr;
        fprintf(file,"unixSeconds,physicalBytes,lifetimePeakPhysicalBytes,compressedBytes,metalAllocatedBytes,ledgerSwapins\n");
        while (!done.load()) {
            task_vm_info_data_t info={}; mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
            if (task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&info,&count)==KERN_SUCCESS) {
                timespec now; clock_gettime(CLOCK_REALTIME,&now);
                fprintf(file,"%.6f,%llu,%lld,%llu,%llu,%lld\n",now.tv_sec+now.tv_nsec/1e9,
                    (unsigned long long)info.phys_footprint,(long long)info.ledger_phys_footprint_peak,
                    (unsigned long long)info.compressed,(unsigned long long)device.currentAllocatedSize,
                    (long long)info.ledger_swapins);
                fflush(file);
            }
            usleep(10000);
        }
        fclose(file);
    }
    return nullptr;
}
__attribute__((constructor)) static void begin() {
    if (getenv("ARENA_MEMORY_DIRECTORY")) active=pthread_create(&thread,nullptr,sample,nullptr)==0;
}
__attribute__((destructor)) static void end() {
    if (active) { done.store(true); pthread_join(thread,nullptr); }
}
