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
    return Bitmap(mtmd_bitmap_init((uint32_t)w,(uint32_t)h,rgb.data()), mtmd_bitmap_free);
}
static const char *grammar = R"GBNF(
root ::= "{" ws "\"clean_pixels\":" ws outcome "," ws "\"authenticity\":" ws outcome "," ws "\"detail\":" ws outcome "," ws "\"quality\":" ws outcome "," ws "\"consistency\":" ws outcome "," ws "\"recommended_source\":" ws outcome "," ws "\"uncertain\":" ws ("true" | "false") "," ws "\"evidence\":" ws "[" ws string ("," ws string)* ws "]" ws "}"
outcome ::= "\"A\"" | "\"B\"" | "\"tie\"" | "\"unknown\""
string ::= "\"" ([^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
ws ::= [ \t\n\r]*
)GBNF";
int main(int argc, char **argv) { @autoreleasepool {
    bool validate = argc == 4 && std::string(argv[1]) == "--validate";
    if (argc != 3 && !validate) { std::cerr << "usage: waifu-photoarena model.gguf mmproj.gguf\n"; return 2; }
    try {
        llama_backend_init();
        auto mp=llama_model_default_params(); mp.n_gpu_layers=99;
        using Model=std::unique_ptr<llama_model,decltype(&llama_model_free)>;
        Model model(llama_model_load_from_file(argv[validate ? 2 : 1],mp),llama_model_free);
        require(bool(model), "Language model failed to load");
        auto cp=llama_context_default_params(); cp.n_ctx=8192; cp.n_batch=512; cp.n_ubatch=512;
        cp.n_threads=4; cp.n_threads_batch=4; cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        using Context=std::unique_ptr<llama_context,decltype(&llama_free)>;
        Context context(llama_init_from_model(model.get(),cp),llama_free);
        require(bool(context), "Decoder failed to initialize");
        auto vp=mtmd_context_params_default(); vp.use_gpu=true; vp.n_threads=4; vp.warmup=false;
        using Vision=std::unique_ptr<mtmd_context,decltype(&mtmd_free)>;
        Vision vision(mtmd_init_from_file(argv[validate ? 3 : 2],model.get(),vp),mtmd_free);
        require(bool(vision) && mtmd_support_vision(vision.get()), "Vision projector failed to load");
        const auto *vocab=llama_model_get_vocab(model.get());
        emit(@{@"ready":@YES,@"backend":@"llama.cpp Metal/CPU",@"compression":@"16x"});
        if (validate) return 0;
        std::string line;
        while (std::getline(std::cin,line)) { @autoreleasepool {
            try {
                require(line.size() <= 48000000, "Request exceeds limit");
                NSData *data=[NSData dataWithBytes:line.data() length:line.size()];
                NSDictionary *request=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
                require([request isKindOfClass:NSDictionary.class], "Invalid JSON request");
                NSArray *images=request[@"images"];
                require([images isKindOfClass:NSArray.class] && images.count==2, "Exactly two images required");
                auto a=bitmap(images[0]), b=bitmap(images[1]);
                require(bool(a) && bool(b), "Image preparation failed");
                const mtmd_bitmap *bitmaps[]={a.get(),b.get()};
                // Exact single-turn, non-thinking suffix of the pinned upstream chat template.
                std::string marker=mtmd_get_marker(vision.get());
                std::string prompt="<|im_start|>system\nYou compare visible photographic evidence. Never infer provenance or authenticity as fact. Return only the requested JSON object.<|im_end|>\n<|im_start|>user\nImage A: "+marker+"\nImage B: "+marker+"\nCompare these two versions of the same scene. All category values name the WINNING SOURCE, never the source containing a defect. For clean_pixels choose the CLEANER image: if A is clean and B has SAMPLE WATERMARK, clean_pixels must be A; if A has SAMPLE WATERMARK and B is clean, clean_pixels must be B. If both inputs are identical, all categories must be tie. for authenticity choose stronger visible natural-source evidence, not beauty; for detail choose recoverable detail; for quality compare blur/compression; for consistency compare scene fidelity. Abstain with unknown when evidence is insufficient. Give short observable evidence. Return JSON keys in order: clean_pixels, authenticity, detail, quality, consistency, recommended_source (each A, B, tie or unknown), uncertain (boolean), evidence (array of strings).<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
                llama_memory_clear(llama_get_memory(context.get()),true);
                using Chunks=std::unique_ptr<mtmd_input_chunks,decltype(&mtmd_input_chunks_free)>;
                Chunks chunks(mtmd_input_chunks_init(),mtmd_input_chunks_free);
                mtmd_input_text text={prompt.c_str(),prompt.size(),true,true};
                auto start=std::chrono::steady_clock::now();
                require(mtmd_tokenize(vision.get(),chunks.get(),&text,bitmaps,2)==0,"Multimodal tokenization failed");
                require(mtmd_helper_get_n_tokens(chunks.get())+512 < 8192,"Image prompt exceeds context budget");
                llama_pos past=0;
                require(mtmd_helper_eval_chunks(vision.get(),context.get(),chunks.get(),0,0,512,true,&past)==0,"Vision/prefill failed");
                auto prefill=std::chrono::steady_clock::now();
                using Sampler=std::unique_ptr<llama_sampler,decltype(&llama_sampler_free)>;
                Sampler sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()),llama_sampler_free);
                auto *constraint=llama_sampler_init_grammar(vocab,grammar,"root");
                require(constraint != nullptr,"JSON grammar failed");
                llama_sampler_chain_add(sampler.get(),constraint);
                llama_sampler_chain_add(sampler.get(),llama_sampler_init_greedy());
                std::string output; int count=0; bool complete=false;
                for (; count<512 && past<8191; count++) {
                    llama_token token=llama_sampler_sample(sampler.get(),context.get(),-1);
                    if (llama_vocab_is_eog(vocab,token)) { complete=true; break; }
                    char piece[256]; int n=llama_token_to_piece(vocab,token,piece,sizeof(piece),0,true);
                    require(n>=0,"Token exceeds output buffer"); output.append(piece,n);
                    llama_batch batch=llama_batch_init(4,0,1); batch.n_tokens=1; batch.token[0]=token;
                    for (int p=0;p<4;p++) batch.pos[p]=past;
                    batch.n_seq_id[0]=1; batch.seq_id[0][0]=0; batch.logits[0]=true;
                    int result=llama_decode(context.get(),batch); llama_batch_free(batch);
                    require(result==0,"Token decode failed"); past++;
                }
                auto end=std::chrono::steady_clock::now();
                NSData *resultData=[NSData dataWithBytes:output.data() length:output.size()];
                NSMutableDictionary *judgment=[[NSJSONSerialization JSONObjectWithData:resultData options:0 error:nil] mutableCopy];
                if (judgment[@"clean_pixels"]) { judgment[@"watermark"] = judgment[@"clean_pixels"]; [judgment removeObjectForKey:@"clean_pixels"]; }
                require(complete && [judgment isKindOfClass:NSDictionary.class],"Model did not finish valid JSON within token budget");
                task_vm_info_data_t info={}; mach_msg_type_number_t size=TASK_VM_INFO_COUNT;
                task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&info,&size);
                emit(@{@"judgment":judgment,@"tokens":@(count),
                       @"prefillSeconds":@(std::chrono::duration<double>(prefill-start).count()),
                       @"decodeSeconds":@(std::chrono::duration<double>(end-prefill).count()),
                       @"physicalFootprintBytes":@(info.phys_footprint),@"compressedBytes":@(info.compressed)});
            } catch (const std::exception &e) { emit(@{@"error":@(e.what())}); }
        }}
    } catch (const std::exception &e) { emit(@{@"error":@(e.what())}); return 1; }
    return 0;
}}
