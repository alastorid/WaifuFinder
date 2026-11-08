#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-metal.h"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cstring>

static void require(bool okay, const char * message) { if (!okay) throw std::runtime_error(message); }
int main() {
    try {
        auto backend=ggml_backend_metal_init(); require(backend,"Metal unavailable");
        auto reg=ggml_backend_metal_reg();
        auto get=(void *(*)(ggml_backend_dev_t))ggml_backend_reg_get_proc_address(reg,"ggml_backend_fusion_get");
        auto enable=(void (*)(void *,bool))ggml_backend_reg_get_proc_address(reg,"ggml_backend_fusion_set_enabled");
        auto init=(void (*)(void *))ggml_backend_reg_get_proc_address(reg,"ggml_backend_fusion_stats_init");
        auto stats=(int (*)(void *,const char **,uint64_t *,int))ggml_backend_reg_get_proc_address(reg,"ggml_backend_fusion_stats_get");
        require(get && enable && init && stats,"Fusion inspection API unavailable");
        auto info=get(ggml_backend_get_device(backend)); init(info);
        int checked=0;
        for (int width: {128,129,1024}) for (int rows: {16,512}) {
            auto ctx=ggml_init({16*1024*1024,nullptr,true}); require(ctx,"Context allocation");
            auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,rows);
            auto w=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,width);
            auto g=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,rows);
            auto norm=ggml_rms_norm(ctx,x,1e-6f);
            auto weighted=ggml_mul(ctx,norm,w);
            auto out=ggml_swiglu_split(ctx,g,weighted);
            auto graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,out);
            auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend); require(buffer,"Buffer allocation");
            std::vector<float> xv(width*rows),gv(xv.size()),wv(width),a(xv.size()),b(xv.size());
            for (size_t i=0;i<xv.size();i++) { xv[i]=std::sin(float(i)*0.019f)*3; gv[i]=std::cos(float(i)*0.003f)*20; }
            for (int i=0;i<width;i++) wv[i]=0.7f+std::sin(float(i)*0.1f)*0.2f;
            ggml_backend_tensor_set(x,xv.data(),0,xv.size()*4);
            ggml_backend_tensor_set(g,gv.data(),0,gv.size()*4);
            ggml_backend_tensor_set(w,wv.data(),0,wv.size()*4);
            enable(info,false); require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"Unfused graph");
            ggml_backend_tensor_get(out,a.data(),0,a.size()*4);
            enable(info,true); require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"Fused graph");
            ggml_backend_tensor_get(out,b.data(),0,b.size()*4);
            for (size_t i=0;i<a.size();i++) require(std::isfinite(b[i]) && std::abs(a[i]-b[i])<=3e-6f*std::max(1.0f,std::abs(a[i])),"Fused/unfused numerical disagreement");
            // Independent F32 reference; GPU reductions may round differently.
            for (int row=0;row<rows;row++) {
                float sum=0; for (int c=0;c<width;c++) { float v=xv[row*width+c]; sum+=v*v; }
                float scale=1.0f/std::sqrt(sum/width+1e-6f);
                for (int c=0;c<width;c++) {
                    int i=row*width+c; float expected=(xv[i]*scale)*wv[c]*(gv[i]/(1+std::exp(-gv[i])));
                    require(std::abs(b[i]-expected)<=1e-5f*std::max(1.0f,std::abs(expected)),"CPU reference disagreement");
                }
            }
            ggml_backend_buffer_free(buffer); ggml_free(ctx); checked++;
        }
        const char * labels[64]; uint64_t counts[64]; int n=stats(info,labels,counts,64); uint64_t fused=0;
        for (int i=0;i<n && i<64;i++) if (std::strstr(labels[i],"RMS_NORM") && std::strstr(labels[i],"GLU")) fused+=counts[i];
        require(fused>=6,"New fused GPU dispatch was not exercised");
        ggml_backend_free(backend);
        std::cout<<"PASS: "<<checked<<" gated RMS shapes, F32 CPU/unfused parity, "<<fused<<" actual fused GPU dispatches\n";
    } catch (const std::exception & e) { std::cerr<<"FAIL: "<<e.what()<<"\n"; return 1; }
}
