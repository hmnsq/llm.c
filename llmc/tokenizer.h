/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
This is all we need for unconditional generation.
If we wanted to later prompt the model, we'd have to add decoding.
Which could be tricky in C because of the regex involved, to look into later.
*/

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <ctype.h>
#ಅನ್ತರ್ಭಾವ <assert.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#ಅನ್ತರ್ಭಾವ "utils.h"

// ----------------------------------------------------------------------------

ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ {
    uint32_t vocab_size;
    ವರ್ಣ **token_table;
    ಪೂರ್ಣಾಙ್ಕ init_ok;
    ಪೂರ್ಣಾಙ್ಕ eot_token; // <|endoftext|> token id
} Tokenizer;

ಶೂನ್ಯ safe_printf(ಧ್ರುವ ವರ್ಣ *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    ಯದಿ (piece == ರಿಕ್ತ) { ಪ್ರತ್ಯಾಗಮ; }
    ಯದಿ (piece[0] == '\0') { ಪ್ರತ್ಯಾಗಮ; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    ಯದಿ (piece[1] == '\0') {
        unsigned ವರ್ಣ byte_val = piece[0];
        ಯದಿ (!(isprint(byte_val) || isspace(byte_val))) {
            ಪ್ರತ್ಯಾಗಮ; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}

ಶೂನ್ಯ tokenizer_init(Tokenizer *tokenizer, ಧ್ರುವ ವರ್ಣ *filename) {
    FILE *file = fopen(filename, "rb");
    ಯದಿ (file == ರಿಕ್ತ) {
        // try to be more helpful as we just added this feature, erase later
        printf("---\n");
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added April 14 2024.\n");
        printf("Re-run `python train_gpt2.py` to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        ಪ್ರತ್ಯಾಗಮ;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, ಪರಿಮಾಣ(uint32_t), 256, file);
    ಪ್ರತಿಜ್ಞಾ(header[0] == 20240328);
    ಪೂರ್ಣಾಙ್ಕ version = header[1];
    tokenizer->vocab_size = header[2];
    ಯದಿ (version == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        ಪ್ರತಿಜ್ಞಾ(tokenizer->vocab_size == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } ಅನ್ಯಥಾ ಯದಿ (version == 2) {
        tokenizer->eot_token = header[3];
    } ಅನ್ಯಥಾ {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, version);
        exit(EXIT_FAILURE);
    }
    // read in all the tokens
    unsigned ವರ್ಣ length;
    tokenizer->token_table = (ವರ್ಣ **)mallocCheck(tokenizer->vocab_size * ಪರಿಮಾಣ(ವರ್ಣ *));
    ಪ್ರತಿ (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        freadCheck(&length, ಪರಿಮಾಣ(unsigned ವರ್ಣ), 1, file);
        ಪ್ರತಿಜ್ಞಾ(length > 0); // every token should be at least one character
        ವರ್ಣ *token_bytes = (ವರ್ಣ *)mallocCheck(length + 1);
        freadCheck(token_bytes, ಪರಿಮಾಣ(ವರ್ಣ), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

ಧ್ರುವ ವರ್ಣ *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    ಯದಿ (tokenizer->init_ok == 0) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }
    ಯದಿ (token_id < tokenizer->vocab_size) {
        ಪ್ರತ್ಯಾಗಮ tokenizer->token_table[token_id];
    } ಅನ್ಯಥಾ {
        printf("invalid token id %u!\n", token_id);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }
}

ಶೂನ್ಯ tokenizer_free(Tokenizer *tokenizer) {
    ಯದಿ (tokenizer->init_ok) {
        ಪ್ರತಿ (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            ಮುಕ್ತ(tokenizer->token_table[i]);
        }
        ಮುಕ್ತ(tokenizer->token_table);
    }
}
