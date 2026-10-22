/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
This is all we need for unconditional generation.
If we wanted to later prompt the model, we'd have to add decoding.
Which could be tricky in C because of the regex involved, to look into later.
*/

#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <ctype.h>
#ଅନ୍ତର୍ଭାଵ <assert.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#ଅନ୍ତର୍ଭାଵ "utils.h"

// ----------------------------------------------------------------------------

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା {
    uint32_t vocab_size;
    ଅକ୍ଷର **token_table;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା init_ok;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା eot_token; // <|endoftext|> token id
} Tokenizer;

ଶୂନ୍ୟ safe_printf(ଅଚଳ ଅକ୍ଷର *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    ଯଦି (piece == ଖାଲି) { ଫେରନ୍ତୁ; }
    ଯଦି (piece[0] == '\0') { ଫେରନ୍ତୁ; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    ଯଦି (piece[1] == '\0') {
        unsigned ଅକ୍ଷର byte_val = piece[0];
        ଯଦି (!(isprint(byte_val) || isspace(byte_val))) {
            ଫେରନ୍ତୁ; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}

ଶୂନ୍ୟ tokenizer_init(Tokenizer *tokenizer, ଅଚଳ ଅକ୍ଷର *filename) {
    FILE *file = fopen(filename, "rb");
    ଯଦି (file == ଖାଲି) {
        // try to be more helpful as we just added this feature, erase later
        printf("---\n");
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added April 14 2024.\n");
        printf("Re-run `python train_gpt2.py` to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        ଫେରନ୍ତୁ;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, ଆକାର(uint32_t), 256, file);
    ପ୍ରତିଜ୍ଞା(header[0] == 20240328);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା version = header[1];
    tokenizer->vocab_size = header[2];
    ଯଦି (version == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        ପ୍ରତିଜ୍ଞା(tokenizer->vocab_size == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } ଅନ୍ୟଥା ଯଦି (version == 2) {
        tokenizer->eot_token = header[3];
    } ଅନ୍ୟଥା {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, version);
        exit(EXIT_FAILURE);
    }
    // read in all the tokens
    unsigned ଅକ୍ଷର length;
    tokenizer->token_table = (ଅକ୍ଷର **)mallocCheck(tokenizer->vocab_size * ଆକାର(ଅକ୍ଷର *));
    ପାଇଁ (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        freadCheck(&length, ଆକାର(unsigned ଅକ୍ଷର), 1, file);
        ପ୍ରତିଜ୍ଞା(length > 0); // every token should be at least one character
        ଅକ୍ଷର *token_bytes = (ଅକ୍ଷର *)mallocCheck(length + 1);
        freadCheck(token_bytes, ଆକାର(ଅକ୍ଷର), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

ଅଚଳ ଅକ୍ଷର *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    ଯଦି (tokenizer->init_ok == 0) {
        ଫେରନ୍ତୁ ଖାଲି;
    }
    ଯଦି (token_id < tokenizer->vocab_size) {
        ଫେରନ୍ତୁ tokenizer->token_table[token_id];
    } ଅନ୍ୟଥା {
        printf("invalid token id %u!\n", token_id);
        ଫେରନ୍ତୁ ଖାଲି;
    }
}

ଶୂନ୍ୟ tokenizer_free(Tokenizer *tokenizer) {
    ଯଦି (tokenizer->init_ok) {
        ପାଇଁ (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            ମୁକ୍ତ(tokenizer->token_table[i]);
        }
        ମୁକ୍ତ(tokenizer->token_table);
    }
}
