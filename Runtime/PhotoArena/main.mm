#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <CoreGraphics/CoreGraphics.h>
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <iostream>
#include <memory>
#include <chrono>
#include <stdexcept>
#include <mach/mach.h>
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>

static void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
static void emit(NSDictionary *value) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:value options:0 error:nil];
    std::cout.write((const char *)json.bytes, json.length); std::cout << '\n' << std::flush;
}
using Bitmap = std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)>;
static Bitmap bitmap(NSString *base64) {
    require([base64 isKindOfClass:NSString.class] && base64.length <= 24000000, "Invalid image payload");
    NSData *data = [[NSData alloc] initWithBase64EncodedString:base64 options:0];
    require(data != nil, "Invalid base64 image");
    CGImageSourceRef source = CGImageSourceCreateWithData((__bridge CFDataRef)data, nullptr);
    require(source != nullptr, "Unreadable image");
    NSDictionary *opts = @{(NSString *)kCGImageSourceCreateThumbnailFromImageAlways:@YES,
                            (NSString *)kCGImageSourceCreateThumbnailWithTransform:@YES,
                            (NSString *)kCGImageSourceThumbnailMaxPixelSize:@1600};
    CGImageRef image = CGImageSourceCreateThumbnailAtIndex(source, 0, (__bridge CFDictionaryRef)opts);
    CFRelease(source); require(image != nullptr, "Image decode failed");
    size_t w = CGImageGetWidth(image), h = CGImageGetHeight(image);
    std::vector<uint8_t> rgba(w*h*4), rgb(w*h*3);
    CGColorSpaceRef color = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGContextRef ctx = CGBitmapContextCreate(rgba.data(), w, h, 8, w*4, color,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(color);
    if (!ctx) { CGImageRelease(image); throw std::runtime_error("Raster allocation failed"); }
    CGContextSetRGBFillColor(ctx, 1, 1, 1, 1); CGContextFillRect(ctx, CGRectMake(0,0,w,h));
    CGContextDrawImage(ctx, CGRectMake(0,0,w,h), image);
    CGContextRelease(ctx); CGImageRelease(image);
    for (size_t i=0;i<w*h;i++) for (int c=0;c<3;c++) rgb[3*i+c]=rgba[4*i+c];
    std::vector<uint8_t>().swap(rgba); // Do not overlap the raster with the owned bitmap copy.
    return Bitmap(mtmd_bitmap_init((uint32_t)w,(uint32_t)h,rgb.data()), mtmd_bitmap_free);
}
static const char *grammar = R"GBNF(
root ::= "{" ws "\"clean_pixels\":" ws outcome "," ws "\"authenticity\":" ws outcome "," ws "\"detail\":" ws outcome "," ws "\"quality\":" ws outcome "," ws "\"consistency\":" ws outcome "," ws "\"recommended_source\":" ws outcome "," ws "\"uncertain\":" ws ("true" | "false") "," ws "\"evidence\":" ws "[" ws string ("," ws string)* ws "]" ws "}"
outcome ::= "\"A\"" | "\"B\"" | "\"tie\"" | "\"unknown\""
string ::= "\"" ([^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
ws ::= [ \t\n\r]*
)GBNF";
static std::string arena_prompt(const std::string & marker) { return "<|im_start|>system\nYou compare visible photographic evidence. Never infer provenance or authenticity as fact. Return only the requested JSON object.<|im_end|>\n<|im_start|>user\nImage A: "+marker+"\nImage B: "+marker+"\nCompare these two versions of the same scene. All category values name the WINNING SOURCE, never the source containing a defect. For clean_pixels choose the CLEANER image: if A is clean and B has SAMPLE WATERMARK, clean_pixels must be A; if A has SAMPLE WATERMARK and B is clean, clean_pixels must be B. If both inputs are identical, all categories must be tie. for authenticity choose stronger visible natural-source evidence, not beauty; for detail choose recoverable detail; for quality compare blur/compression; for consistency compare scene fidelity. Abstain with unknown when evidence is insufficient. Give short observable evidence. Return JSON keys in order: clean_pixels, authenticity, detail, quality, consistency, recommended_source (each A, B, tie or unknown), uncertain (boolean), evidence (array of strings).<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"; }

using Context=std::unique_ptr<llama_context,decltype(&llama_free)>;
using Chunks=std::unique_ptr<mtmd_input_chunks,decltype(&mtmd_input_chunks_free)>;
using Sampler=std::unique_ptr<llama_sampler,decltype(&llama_sampler_free)>;

// Grammar application is a pure candidate filter until accept(). If the global
// greedy winner is legal, filtering the remaining 248,093 tokens cannot change it.
static llama_token legal_greedy(const float * logits, int n_vocab, llama_sampler * constraint) {
    int best=0;
    for (int i=1;i<n_vocab;i++) if (logits[i]>logits[best]) best=i;
    if (std::isfinite(logits[best])) {
        llama_token_data candidate={best,logits[best],0.0f};
        llama_token_data_array single={&candidate,1,-1,false};
        llama_sampler_apply(constraint,&single);
        if (std::isfinite(candidate.logit)) {
            return best;
        }
    }
    return LLAMA_TOKEN_NULL;
}
static llama_token sample_greedy(llama_sampler * sampler, llama_sampler * constraint,
                                  llama_context * context, int n_vocab) {
    const float * logits=llama_get_logits_ith(context,-1);
    require(logits != nullptr,"Missing logits");
    llama_token token=legal_greedy(logits,n_vocab,constraint);
    if (token!=LLAMA_TOKEN_NULL) { llama_sampler_accept(sampler,token); return token; }
    return llama_sampler_sample(sampler,context,-1);
}

struct DecodeBatch {
    llama_batch value=llama_batch_init(4,0,1);
    DecodeBatch() {
        value.n_tokens=1; value.n_seq_id[0]=1;
        value.seq_id[0][0]=0; value.logits[0]=true;
    }
    ~DecodeBatch() { llama_batch_free(value); }
};

int main(int argc, char **argv) { @autoreleasepool {
    bool validate=argc==4 && std::string(argv[1])=="--validate";
    if (argc!=3 && !validate) { std::cerr<<"usage: waifu-photoarena model.gguf mmproj.gguf\n"; return 2; }
    try {
        llama_backend_init();
        auto mp=llama_model_default_params(); mp.n_gpu_layers=99;
        using Model=std::unique_ptr<llama_model,decltype(&llama_model_free)>;
        Model model(llama_model_load_from_file(argv[validate ? 2 : 1],mp),llama_model_free);
        require(bool(model),"Language model failed to load");
        auto cp=llama_context_default_params(); cp.n_ctx=8192; cp.n_batch=512; cp.n_ubatch=512;
        cp.n_threads=4; cp.n_threads_batch=4; cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        auto vp=mtmd_context_params_default(); vp.use_gpu=true; vp.n_threads=4; vp.warmup=false;
        using Vision=std::unique_ptr<mtmd_context,decltype(&mtmd_free)>;
        Vision vision(mtmd_init_from_file(argv[validate ? 3 : 2],model.get(),vp),mtmd_free);
        require(bool(vision) && mtmd_support_vision(vision.get()),"Vision projector failed to load");
        const auto * vocab=llama_model_get_vocab(model.get());
        const int n_vocab=llama_vocab_n_tokens(vocab);
        Context context(nullptr,llama_free);
        if (validate) {
            context.reset(llama_init_from_model(model.get(),cp));
            require(bool(context),"Decoder failed to initialize");
            emit(@{@"ready":@YES}); return 0;
        }
        const std::string marker=mtmd_get_marker(vision.get());
        const std::string prompt=arena_prompt(marker);
        emit(@{@"ready":@YES,@"backend":@"llama.cpp Metal/CPU",@"compression":@"16x",@"pairedComparison":@YES});
        std::string line;
        while (std::getline(std::cin,line)) { @autoreleasepool {
            try {
                require(line.size()<=48000000,"Request exceeds limit");
                const auto preparation_start=std::chrono::steady_clock::now();
                Chunks chunks(mtmd_input_chunks_init(),mtmd_input_chunks_free);
                bool paired=false;
                @autoreleasepool {
                    NSData * data=[NSData dataWithBytesNoCopy:(void *)line.data() length:line.size() freeWhenDone:NO];
                    NSDictionary * request=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
                    require([request isKindOfClass:NSDictionary.class],"Invalid JSON request");
                    NSArray * images=request[@"images"];
                    require([images isKindOfClass:NSArray.class] && images.count==2,"Exactly two images required");
                    auto a=bitmap(images[0]), b=bitmap(images[1]);
                    require(bool(a) && bool(b),"Image preparation failed");
                    mtmd_bitmap_set_id(a.get(),"arena-a"); mtmd_bitmap_set_id(b.get(),"arena-b");
                    paired=[request[@"compareBoth"] isEqual:@YES];
                    const mtmd_bitmap * bitmaps[]={a.get(),b.get()};
                    mtmd_input_text text={prompt.c_str(),prompt.size(),true,true};
                    require(mtmd_tokenize(vision.get(),chunks.get(),&text,bitmaps,2)==0,"Multimodal tokenization failed");
                } // Release decoded RGB, JSON/base64 strings before model compute.
                std::string().swap(line);
                const size_t tokens=mtmd_helper_get_n_tokens(chunks.get());
                require(tokens+512<8192,"Image prompt exceeds context budget");
                std::vector<const mtmd_input_chunk *> order;
                std::vector<size_t> image_slots;
                for (size_t i=0;i<mtmd_input_chunks_size(chunks.get());i++) {
                    auto chunk=mtmd_input_chunks_get(chunks.get(),i); order.push_back(chunk);
                    if (mtmd_input_chunk_get_type(chunk)==MTMD_INPUT_CHUNK_TYPE_IMAGE) image_slots.push_back(i);
                }
                require(image_slots.size()==2,"Expected two complete MiniCPM canvas image chunks");
                // In the pinned MiniCPM-V 4.7 canvas processor each image owns one
                // complete overview/slice chunk. Text, including image-id labels,
                // stays in its original slot. Canvas positions are generated later
                // by mtmd_helper_decode_image_chunk with the new n_past; no KV,
                // recurrent state or absolute M-RoPE positions are reused.
                std::array<std::vector<float>,2> embeddings;
                size_t cached_bytes=0;
                for (int pass=0;pass<(paired ? 2 : 1);pass++) {
                    const auto start=pass ? std::chrono::steady_clock::now() : preparation_start;
                    if (pass) std::swap(order[image_slots[0]],order[image_slots[1]]);
                    // Keep the original token/batch/precision limits; allocate only the
                    // context capacity the complete prompt plus all 512 outputs needs.
                    const uint32_t capacity=(uint32_t)((tokens+512+255)/256*256);
                    double context_init_seconds=0;
                    if (!context || llama_n_ctx(context.get())<capacity) {
                        auto init_start=std::chrono::steady_clock::now();
                        context.reset(); // Never overlap two decoder contexts.
                        cp.n_ctx=capacity;
                        context.reset(llama_init_from_model(model.get(),cp));
                        require(bool(context),"Decoder failed to initialize");
                        context_init_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-init_start).count();
                    }
                    llama_synchronize(context.get());
                    llama_memory_clear(llama_get_memory(context.get()),true);
                    // Cache only within this action. Reserve at most one quarter of
                    // the F16 KV storage removed from the original six full-attention
                    // layers (2 KV heads * 256 dims * K/V * 2 bytes per position).
                    const size_t savings=(8192-llama_n_ctx(context.get()))*size_t(6*2*256*2*2);
                    const size_t cache_limit=paired ? std::min(size_t(8*1024*1024),savings/4) : 0;
                    llama_pos past=0;
                    const auto n_chunks=order.size();
                    for (size_t i=0;i<n_chunks;i++) {
                        const auto * chunk=order[i];
                        if (mtmd_input_chunk_get_type(chunk)==MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                            const char * id=mtmd_input_chunk_get_id(chunk);
                            const int index=id && std::string(id)=="arena-b" ? 1 : 0;
                            const size_t count=mtmd_input_chunk_get_n_tokens(chunk)*size_t(llama_model_n_embd_inp(model.get()));
                            auto & cached=embeddings[index];
                            float * encoded=nullptr;
                            if (pass && cached.size()==count) encoded=cached.data();
                            else {
                                require(mtmd_encode_chunk(vision.get(),chunk)==0,"Vision encode failed");
                                encoded=mtmd_get_output_embd(vision.get());
                                if (!pass && count*sizeof(float)<=cache_limit-cached_bytes) {
                                    cached.assign(encoded,encoded+count); cached_bytes+=count*sizeof(float);
                                }
                            }
                            require(mtmd_helper_decode_image_chunk(vision.get(),context.get(),chunk,encoded,past,0,512,&past,nullptr,nullptr)==0,"Image prefill failed");
                        } else {
                            require(mtmd_helper_eval_chunk_single(vision.get(),context.get(),chunk,past,0,512,i+1==n_chunks,&past)==0,"Text prefill failed");
                        }
                    }
                    if (pass || !paired) { order.clear(); chunks.reset(); }
                    auto prefill=std::chrono::steady_clock::now();
                    Sampler sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()),llama_sampler_free);
                    auto * constraint=llama_sampler_init_grammar(vocab,grammar,"root");
                    require(constraint!=nullptr,"JSON grammar failed");
                    llama_sampler_chain_add(sampler.get(),constraint);
                    llama_sampler_chain_add(sampler.get(),llama_sampler_init_greedy());
                    DecodeBatch batch;
                    std::string output; output.reserve(2048);
                    int count=0; bool complete=false;
                    for (;count<512 && past<8191;count++) {
                        llama_token token=sample_greedy(sampler.get(),constraint,context.get(),n_vocab);
                        if (llama_vocab_is_eog(vocab,token)) { complete=true; break; }
                        char piece[256]; int n=llama_token_to_piece(vocab,token,piece,sizeof(piece),0,true);
                        require(n>=0,"Token exceeds output buffer"); output.append(piece,n);
                        batch.value.token[0]=token;
                        for (int p=0;p<4;p++) batch.value.pos[p]=past;
                        require(llama_decode(context.get(),batch.value)==0,"Token decode failed"); past++;
                    }
                    auto end=std::chrono::steady_clock::now();
                    NSData * resultData=[NSData dataWithBytes:output.data() length:output.size()];
                    NSMutableDictionary * judgment=[[NSJSONSerialization JSONObjectWithData:resultData options:0 error:nil] mutableCopy];
                    require(complete && [judgment isKindOfClass:NSDictionary.class],"Model did not finish valid JSON within token budget");
                    if (judgment[@"clean_pixels"]) { judgment[@"watermark"]=judgment[@"clean_pixels"]; [judgment removeObjectForKey:@"clean_pixels"]; }
                    task_vm_info_data_t info={}; mach_msg_type_number_t size=TASK_VM_INFO_COUNT;
                    task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&info,&size);
                    emit(@{@"judgment":judgment,@"tokens":@(count),@"promptTokens":@(tokens),
                        @"contextCapacity":@(llama_n_ctx(context.get())),@"embeddingCacheBytes":@(cached_bytes),
                        @"contextInitSeconds":@(context_init_seconds),
                        @"prefillSeconds":@(std::chrono::duration<double>(prefill-start).count()-context_init_seconds),
                        @"decodeSeconds":@(std::chrono::duration<double>(end-prefill).count()),
                        @"physicalFootprintBytes":@(info.phys_footprint),@"compressedBytes":@(info.compressed)});
                }
                // No cross-action model state or image embedding cache. The app
                // closes this worker after the reconciled comparison.
                llama_synchronize(context.get());
                llama_memory_clear(llama_get_memory(context.get()),true);
            } catch (const std::exception & e) {
                context.reset(); // Failed jobs cannot leave stale recurrent state.
                emit(@{@"error":@(e.what())});
            }
        }}
    } catch (const std::exception & e) { emit(@{@"error":@(e.what())}); return 1; }
    return 0;
}}
