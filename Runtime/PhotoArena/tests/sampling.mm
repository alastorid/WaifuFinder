// The actual worker helper, not a copied implementation.
#define main photoarena_worker_main
#include "../main.mm"
#undef main
int main(int argc,char **argv) { @autoreleasepool {
    try {
        require(argc==2,"Pass the pinned model.gguf for vocabulary-only correctness");
        llama_backend_init();
        auto mp=llama_model_default_params(); mp.vocab_only=true; mp.n_gpu_layers=0;
        auto model=llama_model_load_from_file(argv[1],mp); require(model,"Vocabulary load failed");
        const auto * vocab=llama_model_get_vocab(model); int size=llama_vocab_n_tokens(vocab);
        Sampler reference(llama_sampler_init_grammar(vocab,grammar,"root"),llama_sampler_free);
        Sampler optimized(llama_sampler_init_grammar(vocab,grammar,"root"),llama_sampler_free);
        Sampler greedy(llama_sampler_init_greedy(),llama_sampler_free);
        const std::string output=R"({"clean_pixels":"A","authenticity":"unknown","detail":"tie","quality":"B","consistency":"unknown","recommended_source":"A","uncertain":true,"evidence":["Visible text","Quote: \"sample\""]})";
        std::vector<llama_token> tokens(output.size()+16);
        int n=llama_tokenize(vocab,output.data(),output.size(),tokens.data(),tokens.size(),false,false); require(n>0,"Tokenization failed");
        std::vector<float> logits(size,-20);
        std::vector<llama_token_data> full(size);
        int fast_count=0,fallback_count=0;
        for (int step=0;step<n;step++) {
            std::fill(logits.begin(),logits.end(),-20);
            logits[tokens[step]]=10;
            // Force a rejected raw argmax on alternating steps, without changing
            // which token the original full grammar+greedy chain must select.
            if (step%2) for (int candidate=0;candidate<size;candidate++) {
                llama_token_data one={candidate,20,0}; llama_token_data_array view={&one,1,-1,false};
                llama_sampler_apply(reference.get(),&view);
                if (!std::isfinite(one.logit)) { logits[candidate]=20; break; }
            }
            for (int i=0;i<size;i++) full[i]={i,logits[i],0};
            llama_token_data_array view={full.data(),full.size(),-1,false};
            llama_sampler_apply(reference.get(),&view); llama_sampler_apply(greedy.get(),&view);
            llama_token expected=view.data[view.selected].id;
            llama_token actual=legal_greedy(logits.data(),size,optimized.get());
            if (actual==LLAMA_TOKEN_NULL) {
                fallback_count++;
                for (int i=0;i<size;i++) full[i]={i,logits[i],0};
                view={full.data(),full.size(),-1,false};
                llama_sampler_apply(optimized.get(),&view); llama_sampler_apply(greedy.get(),&view);
                actual=view.data[view.selected].id;
            } else fast_count++;
            require(actual==expected && actual==tokens[step],"Grammar fast/fallback token disagreement");
            llama_sampler_accept(reference.get(),expected); llama_sampler_accept(optimized.get(),actual);
        }
        require(fast_count>0 && fallback_count>0,"Both sampling paths must be exercised");
        llama_model_free(model);
        std::cout<<"PASS: "<<n<<" exact grammar token decisions, legal-argmax and full-vocabulary fallback\n";
    } catch(const std::exception &e) { std::cerr<<"FAIL: "<<e.what()<<"\n";return 1; }
}}
