// Verify preprocessing/chunk reuse against an independently tokenized reverse
// order, including every canvas M-RoPE coordinate. Does not run model inference.
#define main photoarena_worker_main
#include "../main.mm"
#undef main
int main(int argc,char **argv) { @autoreleasepool {
    try {
        require(argc==5,"Pass model, projector, first image and differently shaped second image");
        llama_backend_init();
        auto mp=llama_model_default_params(); mp.n_gpu_layers=99;
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
        require(bool(model),"Model load failed");
        auto params=mtmd_context_params_default(); params.use_gpu=true; params.n_threads=4; params.warmup=false;
        std::unique_ptr<mtmd_context,decltype(&mtmd_free)> vision(mtmd_init_from_file(argv[2],model.get(),params),mtmd_free);
        require(bool(vision),"Projector load failed");
        NSData * ad=[NSData dataWithContentsOfFile:@(argv[3])], * bd=[NSData dataWithContentsOfFile:@(argv[4])];
        auto a=bitmap([ad base64EncodedStringWithOptions:0]), b=bitmap([bd base64EncodedStringWithOptions:0]);
        mtmd_bitmap_set_id(a.get(),"arena-a"); mtmd_bitmap_set_id(b.get(),"arena-b");
        const auto prompt=arena_prompt(mtmd_get_marker(vision.get()));
        mtmd_input_text text={prompt.c_str(),prompt.size(),true,true};
        Chunks forward(mtmd_input_chunks_init(),mtmd_input_chunks_free),reverse(mtmd_input_chunks_init(),mtmd_input_chunks_free);
        const mtmd_bitmap * ab[]={a.get(),b.get()}, * ba[]={b.get(),a.get()};
        require(mtmd_tokenize(vision.get(),forward.get(),&text,ab,2)==0,"Forward tokenization");
        require(mtmd_tokenize(vision.get(),reverse.get(),&text,ba,2)==0,"Reverse tokenization");
        size_t n=mtmd_input_chunks_size(forward.get()); require(n==mtmd_input_chunks_size(reverse.get()),"Chunk count");
        std::vector<const mtmd_input_chunk *> order; std::vector<size_t> slots;
        for (size_t i=0;i<n;i++) { auto c=mtmd_input_chunks_get(forward.get(),i); order.push_back(c); if(mtmd_input_chunk_get_type(c)==MTMD_INPUT_CHUNK_TYPE_IMAGE)slots.push_back(i); }
        require(slots.size()==2,"Two canvas chunks"); std::swap(order[slots[0]],order[slots[1]]);
        llama_pos past=0;size_t positions=0;
        for(size_t i=0;i<n;i++) {
            auto reused=order[i],fresh=mtmd_input_chunks_get(reverse.get(),i);
            require(mtmd_input_chunk_get_type(reused)==mtmd_input_chunk_get_type(fresh),"Chunk type");
            require(mtmd_input_chunk_get_n_tokens(reused)==mtmd_input_chunk_get_n_tokens(fresh),"Token count");
            require(mtmd_input_chunk_get_n_pos(reused)==mtmd_input_chunk_get_n_pos(fresh),"Position count");
            if(mtmd_input_chunk_get_type(reused)==MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t na,nb; auto at=mtmd_input_chunk_get_tokens_text(reused,&na),bt=mtmd_input_chunk_get_tokens_text(fresh,&nb);
                require(na==nb && std::equal(at,at+na,bt),"Reversed text/role tokens differ");
            }else{
                require(std::string(mtmd_input_chunk_get_id(reused))==mtmd_input_chunk_get_id(fresh),"Image identity");
                auto at=mtmd_input_chunk_get_tokens_image(reused),bt=mtmd_input_chunk_get_tokens_image(fresh);
                for(size_t j=0;j<mtmd_input_chunk_get_n_tokens(reused);j++) {
                    auto ap=mtmd_image_tokens_get_decoder_pos(at,past,j),bp=mtmd_image_tokens_get_decoder_pos(bt,past,j);
                    require(ap.t==bp.t && ap.x==bp.x && ap.y==bp.y && ap.z==bp.z,"Reversed M-RoPE coordinates differ");positions++;
                }
            }
            past+=mtmd_input_chunk_get_n_pos(reused);
        }
        std::cout<<"PASS: reversed role tokens and "<<positions<<" canvas decoder positions equal fresh preprocessing\n";
    }catch(const std::exception &e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
}}
