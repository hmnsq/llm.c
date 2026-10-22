/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
This is all we need for unconditional generation.
If we wanted to later prompt the model, we'd have to add decoding.
Which could be tricky in C because of the regex involved, to look into later.
*/

#include <stdint.h>
#include <ctype.h>
#include <assert.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "utils.h"

// ----------------------------------------------------------------------------

typedef sa_nracanaa {
    uint32_t vocab_size;
    var_na **token_table;
    ཧྲིལ init_ok;
    ཧྲིལ eot_token; // <|endoftext|> token id
} Tokenizer;

void safe_printf(const var_na *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    གལ་སྲིད (piece == NULL) { ལོག; }
    གལ་སྲིད (piece[0] == '\0') { ལོག; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    གལ་སྲིད (piece[1] == '\0') {
        unsigned var_na byte_val = piece[0];
        གལ་སྲིད (!(isprint(byte_val) || isspace(byte_val))) {
            ལོག; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}

void tokenizer_init(Tokenizer *tokenizer, const var_na *filename) {
    FILE *file = fopen(filename, "rb");
    གལ་སྲིད (file == NULL) {
        // try to be more helpful as we just added this feature, erase later
        printf("---\n");
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added April 14 2024.\n");
        printf("Re-run `python train_gpt2.py` to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        ལོག;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, parimaa_na(uint32_t), 256, file);
    assert(header[0] == 20240328);
    ཧྲིལ version = header[1];
    tokenizer->vocab_size = header[2];
    གལ་སྲིད (version == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        assert(tokenizer->vocab_size == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } དེ་མིན གལ་སྲིད (version == 2) {
        tokenizer->eot_token = header[3];
    } དེ་མིན {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, version);
        exit(EXIT_FAILURE);
    }
    // read in all the tokens
    unsigned var_na length;
    tokenizer->token_table = (var_na **)mallocCheck(tokenizer->vocab_size * parimaa_na(var_na *));
    དོན་ལུ (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        freadCheck(&length, parimaa_na(unsigned var_na), 1, file);
        assert(length > 0); // every token should be at least one character
        var_na *token_bytes = (var_na *)mallocCheck(length + 1);
        freadCheck(token_bytes, parimaa_na(var_na), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

const var_na *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    གལ་སྲིད (tokenizer->init_ok == 0) {
        ལོག NULL;
    }
    གལ་སྲིད (token_id < tokenizer->vocab_size) {
        ལོག tokenizer->token_table[token_id];
    } དེ་མིན {
        printf("invalid token id %u!\n", token_id);
        ལོག NULL;
    }
}

void tokenizer_free(Tokenizer *tokenizer) {
    གལ་སྲིད (tokenizer->init_ok) {
        དོན་ལུ (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            free(tokenizer->token_table[i]);
        }
        free(tokenizer->token_table);
    }
}
