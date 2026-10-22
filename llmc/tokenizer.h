/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
This is all we need for unconditional generation.
If we wanted to later prompt the model, we'd have to add decoding.
Which could be tricky in C because of the regex involved, to look into later.
*/

#ضمن <stdint.h>
#ضمن <ctype.h>
#ضمن <assert.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#ضمن "utils.h"

// ----------------------------------------------------------------------------

تعريفنوع بنية {
    uint32_t vocab_size;
    حرف **token_table;
    صحيح init_ok;
    صحيح eot_token; // <|endoftext|> token id
} Tokenizer;

فراغ safe_printf(ثابت حرف *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    𐤀𐤌 (piece == عدم) { أرجع; }
    𐤀𐤌 (piece[0] == '\0') { أرجع; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    𐤀𐤌 (piece[1] == '\0') {
        unsigned حرف byte_val = piece[0];
        𐤀𐤌 (!(isprint(byte_val) || isspace(byte_val))) {
            أرجع; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}

فراغ tokenizer_init(Tokenizer *tokenizer, ثابت حرف *filename) {
    FILE *file = fopen(filename, "rb");
    𐤀𐤌 (file == عدم) {
        // try to be more helpful as we just added this feature, erase later
        printf("---\n");
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added April 14 2024.\n");
        printf("Re-run `python train_gpt2.py` to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        أرجع;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, حجم(uint32_t), 256, file);
    أكد(header[0] == 20240328);
    صحيح version = header[1];
    tokenizer->vocab_size = header[2];
    𐤀𐤌 (version == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        أكد(tokenizer->vocab_size == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } وإلا 𐤀𐤌 (version == 2) {
        tokenizer->eot_token = header[3];
    } وإلا {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, version);
        exit(EXIT_FAILURE);
    }
    // read in all the tokens
    unsigned حرف length;
    tokenizer->token_table = (حرف **)mallocCheck(tokenizer->vocab_size * حجم(حرف *));
    لكل (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        freadCheck(&length, حجم(unsigned حرف), 1, file);
        أكد(length > 0); // every token should be at least one character
        حرف *token_bytes = (حرف *)mallocCheck(length + 1);
        freadCheck(token_bytes, حجم(حرف), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

ثابت حرف *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    𐤀𐤌 (tokenizer->init_ok == 0) {
        أرجع عدم;
    }
    𐤀𐤌 (token_id < tokenizer->vocab_size) {
        أرجع tokenizer->token_table[token_id];
    } وإلا {
        printf("invalid token id %u!\n", token_id);
        أرجع عدم;
    }
}

فراغ tokenizer_free(Tokenizer *tokenizer) {
    𐤀𐤌 (tokenizer->init_ok) {
        لكل (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            حرر(tokenizer->token_table[i]);
        }
        حرر(tokenizer->token_table);
    }
}
