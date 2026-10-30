#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include "llama.h"

#define TAG "AOT"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static llama_model * g_model = nullptr;
static int g_n_ctx = 2048;
static int g_n_threads = 4;
static bool g_backend_ready = false;

// Độ dài phần đầu chuỗi là UTF-8 hoàn chỉnh (tránh cắt giữa ký tự tiếng Việt/emoji)
static size_t utf8_complete_len(const std::string & s) {
    size_t n = s.size();
    for (size_t back = 1; back <= 4 && back <= n; back++) {
        unsigned char c = (unsigned char) s[n - back];
        if ((c & 0xC0) == 0x80) continue;
        size_t need = 1;
        if ((c & 0xE0) == 0xC0) need = 2;
        else if ((c & 0xF0) == 0xE0) need = 3;
        else if ((c & 0xF8) == 0xF0) need = 4;
        if (need > back) return n - back;
        return n;
    }
    return n;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mycompany_aot_LlamaEngine_nativeLoad(JNIEnv * env, jobject, jstring jpath, jint nCtx, jint nThreads) {
    const char * path = env->GetStringUTFChars(jpath, nullptr);
    if (!g_backend_ready) {
        llama_backend_init();
        g_backend_ready = true;
    }
    g_n_ctx = nCtx;
    g_n_threads = nThreads;

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    g_model = llama_model_load_from_file(path, mp);
    env->ReleaseStringUTFChars(jpath, path);

    if (!g_model) LOGE("Khong load duoc model");
    return g_model != nullptr;
}

// Trả về: 0 = ok, -1 = prompt quá dài, -2 = lỗi
extern "C" JNIEXPORT jint JNICALL
Java_com_mycompany_aot_LlamaEngine_nativeGenerate(JNIEnv * env, jobject, jbyteArray jprompt, jint maxTokens, jobject cb) {
    if (!g_model) return -2;

    jsize plen = env->GetArrayLength(jprompt);
    std::string prompt(plen, '\0');
    env->GetByteArrayRegion(jprompt, 0, plen, (jbyte *) &prompt[0]);

    const llama_vocab * vocab = llama_model_get_vocab(g_model);

    int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) return -2;
    if (n_prompt >= g_n_ctx - 16) return -1;

    std::vector<llama_token> toks(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), toks.data(), (int32_t) toks.size(), true, true) < 0) {
        return -2;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = g_n_ctx;
    cp.n_batch = g_n_ctx;
    cp.n_threads = g_n_threads;
    cp.n_threads_batch = g_n_threads;
    llama_context * ctx = llama_init_from_model(g_model, cp);
    if (!ctx) return -2;

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, 1.1f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    jclass cls = env->GetObjectClass(cb);
    jmethodID mid = env->GetMethodID(cls, "onToken", "([B)Z");

    llama_token tok = 0;
    llama_batch batch = llama_batch_get_one(toks.data(), (int32_t) toks.size());
    int n_pos = 0;
    bool stop = false;
    std::string pending;

    for (int i = 0; i < maxTokens && !stop; i++) {
        if (n_pos + batch.n_tokens > g_n_ctx) break;
        if (llama_decode(ctx, batch) != 0) break;
        n_pos += batch.n_tokens;

        tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
        if (n < 0) break;
        pending.append(buf, n);

        size_t k = utf8_complete_len(pending);
        if (k > 0) {
            jbyteArray arr = env->NewByteArray((jsize) k);
            env->SetByteArrayRegion(arr, 0, (jsize) k, (const jbyte *) pending.data());
            pending.erase(0, k);
            jboolean cont = env->CallBooleanMethod(cb, mid, arr);
            env->DeleteLocalRef(arr);
            if (!cont) stop = true;
        }

        batch = llama_batch_get_one(&tok, 1);
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    return 0;
}

extern "C" JNIEXPORT void JNICALL
Java_com_mycompany_aot_LlamaEngine_nativeFree(JNIEnv *, jobject) {
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
}
