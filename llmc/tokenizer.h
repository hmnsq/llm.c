/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
This is all we need for unconditional generation.
If we wanted to later prompt the model, we'd have to add decoding.
Which could be tricky in C because of the regex involved, to look into later.
*/

#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <ctype.h>
#അന്തര്ഭാവ <assert.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#അന്തര്ഭാവ "utils.h"

// ----------------------------------------------------------------------------

തരനിർവചനം ഘടന {
    uint32_t vocab_size;
    അക്ഷരം **token_table;
    പൂർണ്ണസംഖ്യ init_ok;
    പൂർണ്ണസംഖ്യ eot_token; // <|endoftext|> token id
} Tokenizer;

ശൂന്യം safe_printf(സ്ഥിരം അക്ഷരം *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    എങ്കിൽ (piece == ശൂന്യ) { തിരികെ; }
    എങ്കിൽ (piece[0] == '\0') { തിരികെ; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    എങ്കിൽ (piece[1] == '\0') {
        unsigned അക്ഷരം byte_val = piece[0];
        എങ്കിൽ (!(isprint(byte_val) || isspace(byte_val))) {
            തിരികെ; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}

ശൂന്യം tokenizer_init(Tokenizer *tokenizer, സ്ഥിരം അക്ഷരം *filename) {
    FILE *file = fopen(filename, "rb");
    എങ്കിൽ (file == ശൂന്യ) {
        // try to be more helpful as we just added this feature, erase later
        printf("---\n");
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added April 14 2024.\n");
        printf("Re-run `python train_gpt2.py` to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        തിരികെ;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, വലുപ്പം(uint32_t), 256, file);
    പ്രതിജ്ഞാ(header[0] == 20240328);
    പൂർണ്ണസംഖ്യ version = header[1];
    tokenizer->vocab_size = header[2];
    എങ്കിൽ (version == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        പ്രതിജ്ഞാ(tokenizer->vocab_size == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } അല്ലെങ്കിൽ എങ്കിൽ (version == 2) {
        tokenizer->eot_token = header[3];
    } അല്ലെങ്കിൽ {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, version);
        exit(EXIT_FAILURE);
    }
    // read in all the tokens
    unsigned അക്ഷരം length;
    tokenizer->token_table = (അക്ഷരം **)mallocCheck(tokenizer->vocab_size * വലുപ്പം(അക്ഷരം *));
    വേണ്ടി (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        freadCheck(&length, വലുപ്പം(unsigned അക്ഷരം), 1, file);
        പ്രതിജ്ഞാ(length > 0); // every token should be at least one character
        അക്ഷരം *token_bytes = (അക്ഷരം *)mallocCheck(length + 1);
        freadCheck(token_bytes, വലുപ്പം(അക്ഷരം), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

സ്ഥിരം അക്ഷരം *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    എങ്കിൽ (tokenizer->init_ok == 0) {
        തിരികെ ശൂന്യ;
    }
    എങ്കിൽ (token_id < tokenizer->vocab_size) {
        തിരികെ tokenizer->token_table[token_id];
    } അല്ലെങ്കിൽ {
        printf("invalid token id %u!\n", token_id);
        തിരികെ ശൂന്യ;
    }
}

ശൂന്യം tokenizer_free(Tokenizer *tokenizer) {
    എങ്കിൽ (tokenizer->init_ok) {
        വേണ്ടി (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            സ്വതന്ത്രം(tokenizer->token_table[i]);
        }
        സ്വതന്ത്രം(tokenizer->token_table);
    }
}
