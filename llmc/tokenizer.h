/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
This is all we need for unconditional generation.
If we wanted to later prompt the model, we'd have to add decoding.
Which could be tricky in C because of the regex involved, to look into later.
*/

#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <ctype.h>
#அந்தர்பாவ <assert.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#அந்தர்பாவ "utils.h"

// ----------------------------------------------------------------------------

வகைவரை கட்டமைப்பு {
    uint32_t vocab_size;
    எழுத்து **token_table;
    முழு init_ok;
    முழு eot_token; // <|endoftext|> token id
} Tokenizer;

வெறுமை safe_printf(மாறிலி எழுத்து *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    எனில் (piece == வெற்று) { திரும்பு; }
    எனில் (piece[0] == '\0') { திரும்பு; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    எனில் (piece[1] == '\0') {
        unsigned எழுத்து byte_val = piece[0];
        எனில் (!(isprint(byte_val) || isspace(byte_val))) {
            திரும்பு; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}

வெறுமை tokenizer_init(Tokenizer *tokenizer, மாறிலி எழுத்து *filename) {
    FILE *file = fopen(filename, "rb");
    எனில் (file == வெற்று) {
        // try to be more helpful as we just added this feature, erase later
        printf("---\n");
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added April 14 2024.\n");
        printf("Re-run `python train_gpt2.py` to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        திரும்பு;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, அளவு(uint32_t), 256, file);
    ப்ரதிஜ்ஞா(header[0] == 20240328);
    முழு version = header[1];
    tokenizer->vocab_size = header[2];
    எனில் (version == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        ப்ரதிஜ்ஞா(tokenizer->vocab_size == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } இல்லையெனில் எனில் (version == 2) {
        tokenizer->eot_token = header[3];
    } இல்லையெனில் {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, version);
        exit(EXIT_FAILURE);
    }
    // read in all the tokens
    unsigned எழுத்து length;
    tokenizer->token_table = (எழுத்து **)mallocCheck(tokenizer->vocab_size * அளவு(எழுத்து *));
    க்கு (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        freadCheck(&length, அளவு(unsigned எழுத்து), 1, file);
        ப்ரதிஜ்ஞா(length > 0); // every token should be at least one character
        எழுத்து *token_bytes = (எழுத்து *)mallocCheck(length + 1);
        freadCheck(token_bytes, அளவு(எழுத்து), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

மாறிலி எழுத்து *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    எனில் (tokenizer->init_ok == 0) {
        திரும்பு வெற்று;
    }
    எனில் (token_id < tokenizer->vocab_size) {
        திரும்பு tokenizer->token_table[token_id];
    } இல்லையெனில் {
        printf("invalid token id %u!\n", token_id);
        திரும்பு வெற்று;
    }
}

வெறுமை tokenizer_free(Tokenizer *tokenizer) {
    எனில் (tokenizer->init_ok) {
        க்கு (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            விடுவி(tokenizer->token_table[i]);
        }
        விடுவி(tokenizer->token_table);
    }
}
