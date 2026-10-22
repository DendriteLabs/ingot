// Teacher-forced reads of a gguf model with the pinned llama-prism fork: for each request, the
// hidden states after the requested layers at every row from `first` on, written as the same
// safetensors file a vLLM server writes for the eval (hidden_states [rows, layers, n_embd],
// token_ids [rows]; rows before `first` are zero). With "logits": true the raw next-token logits
// of those rows are written too (logits [rows - first, n_vocab]).
//
// One process per GPU. Requests are JSON lines on stdin, one JSON line answers each on stdout:
//   {"tokens": [...], "first": 123, "out": "/dev/shm/x.safetensors", "logits": false}
//   -> {"ok": true} | {"ok": false, "error": "..."}
//
//   llama-states -m model.gguf --layers 7,15,...,63 [-c 32768] [-b 2048]
// Layers are the fork's 0-based block indices: block il's output is HF hidden_states[il + 1].

#include "llama.h"
#include "../src/llama-ext.h"

#include <nlohmann/json.hpp>

#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

struct Args {
    std::string model;
    std::vector<int32_t> layers;
    int n_ctx = 32768;
    int n_batch = 2048;
};

static Args parse(int argc, char ** argv) {
    Args a;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "-m") {
            a.model = v;
        } else if (k == "-c") {
            a.n_ctx = std::stoi(v);
        } else if (k == "-b") {
            a.n_batch = std::stoi(v);
        } else if (k == "--layers") {
            size_t pos = 0;
            while (pos < v.size()) {
                size_t comma = v.find(',', pos);
                a.layers.push_back(std::stoi(v.substr(pos, comma - pos)));
                pos = comma == std::string::npos ? v.size() : comma + 1;
            }
        } else {
            throw std::invalid_argument("unknown flag " + k);
        }
    }
    if (a.model.empty() || a.layers.empty()) {
        throw std::invalid_argument("usage: llama-states -m model.gguf --layers 7,15,... [-c N] [-b N]");
    }
    return a;
}

// A minimal safetensors writer: F32 / I64 tensors, data back to back.
struct Tensor {
    std::string name, dtype;
    std::vector<int64_t> shape;
    const void * data;
    size_t bytes;
};

static void write_safetensors(const std::string & path, const std::vector<Tensor> & tensors) {
    json head = json::object();
    size_t at = 0;
    for (const auto & t : tensors) {
        head[t.name] = {{"dtype", t.dtype}, {"shape", t.shape}, {"data_offsets", {at, at + t.bytes}}};
        at += t.bytes;
    }
    std::string h = head.dump();
    h.append((8 - h.size() % 8) % 8, ' ');
    // the eval reads a file only after taking the shared lock on <path>.lock, which is held
    // exclusively here until the file is complete
    int lock = open((path + ".lock").c_str(), O_CREAT | O_RDWR, 0644);
    if (lock < 0 || flock(lock, LOCK_EX) != 0) {
        throw std::runtime_error("cannot lock " + path + ".lock");
    }
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        close(lock);
        throw std::runtime_error("cannot write " + path);
    }
    uint64_t n = h.size();
    fwrite(&n, 8, 1, f);
    fwrite(h.data(), 1, h.size(), f);
    for (const auto & t : tensors) {
        fwrite(t.data, 1, t.bytes, f);
    }
    fclose(f);
    flock(lock, LOCK_UN);
    close(lock);
}

static void read(llama_context * ctx, const Args & a, int n_embd, int n_vocab, const json & req) {
    const auto tokens = req.at("tokens").get<std::vector<llama_token>>();
    const int first = req.at("first").get<int>();
    const bool want_logits = req.value("logits", false);
    const int rows = (int) tokens.size();
    if (rows < 1 || rows > a.n_ctx || first < 0 || first >= rows) {
        throw std::invalid_argument("tokens/first out of range");
    }
    const int n_layers = (int) a.layers.size();
    // rows before `first` are prompt rows the eval never reads: not computed out, not written
    std::vector<float> states((size_t) (rows - first) * n_layers * n_embd);
    std::vector<float> logits(want_logits ? (size_t) (rows - first) * n_vocab : 0);

    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_init(a.n_batch, 0, 1);
    for (int start = 0; start < rows; start += a.n_batch) {
        const int n = std::min(a.n_batch, rows - start);
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            batch.token[i] = tokens[start + i];
            batch.pos[i] = start + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = start + i >= first;
        }
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("llama_decode failed");
        }
        for (int i = 0; i < n; ++i) {
            const int r = start + i;
            if (r < first) {
                continue;
            }
            const float * cap = llama_get_embeddings_capture_ith(ctx, i);
            if (!cap) {
                llama_batch_free(batch);
                throw std::runtime_error("no captured states");
            }
            std::memcpy(&states[(size_t) (r - first) * n_layers * n_embd], cap,
                        sizeof(float) * n_layers * n_embd);
            if (want_logits) {
                std::memcpy(&logits[(size_t) (r - first) * n_vocab], llama_get_logits_ith(ctx, i),
                            sizeof(float) * n_vocab);
            }
        }
    }
    llama_batch_free(batch);

    std::vector<int64_t> ids(tokens.begin(), tokens.end());
    int64_t first_row = first;
    std::vector<Tensor> out = {
        {"hidden_states", "F32", {rows - first, n_layers, n_embd}, states.data(),
         states.size() * sizeof(float)},
        {"token_ids", "I64", {rows}, ids.data(), ids.size() * sizeof(int64_t)},
        {"first_row", "I64", {1}, &first_row, sizeof(int64_t)},
    };
    if (want_logits) {
        out.push_back({"logits", "F32", {rows - first, n_vocab}, logits.data(), logits.size() * sizeof(float)});
    }
    write_safetensors(req.at("out").get<std::string>(), out);
}

int main(int argc, char ** argv) {
    Args a;
    try {
        a = parse(argc, argv);
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 2;
    }
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(a.model.c_str(), mparams);
    if (!model) {
        std::cerr << "cannot load " << a.model << "\n";
        return 1;
    }
    auto cparams = llama_context_default_params();
    cparams.n_ctx = a.n_ctx;
    cparams.n_batch = a.n_batch;
    cparams.n_ubatch = a.n_batch;
    cparams.n_seq_max = 1;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    // the KV cache stays 16-bit, as the teacher's and every fp8 model's in vLLM
    cparams.type_k = GGML_TYPE_F16;
    cparams.type_v = GGML_TYPE_F16;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::cerr << "cannot create a context\n";
        return 1;
    }
    llama_set_capture_layers(ctx, a.layers.data(), a.layers.size());
    const int n_embd = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::cout << json{{"ready", true}, {"n_embd", n_embd}, {"n_vocab", n_vocab}}.dump() << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        json reply;
        try {
            read(ctx, a, n_embd, n_vocab, json::parse(line));
            reply = {{"ok", true}};
        } catch (const std::exception & e) {
            reply = {{"ok", false}, {"error", e.what()}};
        }
        std::cout << reply.dump() << std::endl;
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
