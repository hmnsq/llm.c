/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
This is all we need for unconditional generation.
If we wanted to later prompt the model, we'd have to add decoding.
Which could be tricky in C because of the regex involved, to look into later.
*/

#شامل <stdint.h>
#شامل <ctype.h>
#شامل <assert.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#شامل "utils.h"

// ----------------------------------------------------------------------------

تعریفنوع ساختار {
    uint32_t vocab_size;
    نویسه **token_table;
    صحیح init_ok;
    صحیح eot_token; // <|endoftext|> token id
} Tokenizer;

تهی safe_printf(ثابت نویسه *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    اگر (piece == پوچ) { بازگشت; }
    اگر (piece[0] == '\0') { بازگشت; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    اگر (piece[1] == '\0') {
        unsigned نویسه byte_val = piece[0];
        اگر (!(isprint(byte_val) || isspace(byte_val))) {
            بازگشت; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}

تهی tokenizer_init(Tokenizer *tokenizer, ثابت نویسه *filename) {
    FILE *file = fopen(filename, "rb");
    اگر (file == پوچ) {
        // try to be more helpful as we just added this feature, erase later
        printf("---\n");
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added April 14 2024.\n");
        printf("Re-run `python train_gpt2.py` to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        بازگشت;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, اندازه(uint32_t), 256, file);
    أكد(header[0] == 20240328);
    صحیح version = header[1];
    tokenizer->vocab_size = header[2];
    اگر (version == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        أكد(tokenizer->vocab_size == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } وگرنه اگر (version == 2) {
        tokenizer->eot_token = header[3];
    } وگرنه {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, version);
        exit(EXIT_FAILURE);
    }
    // read in all the tokens
    unsigned نویسه length;
    tokenizer->token_table = (نویسه **)mallocCheck(tokenizer->vocab_size * اندازه(نویسه *));
    برای (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        freadCheck(&length, اندازه(unsigned نویسه), 1, file);
        أكد(length > 0); // every token should be at least one character
        نویسه *token_bytes = (نویسه *)mallocCheck(length + 1);
        freadCheck(token_bytes, اندازه(نویسه), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

ثابت نویسه *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    اگر (tokenizer->init_ok == 0) {
        بازگشت پوچ;
    }
    اگر (token_id < tokenizer->vocab_size) {
        بازگشت tokenizer->token_table[token_id];
    } وگرنه {
        printf("invalid token id %u!\n", token_id);
        بازگشت پوچ;
    }
}

تهی tokenizer_free(Tokenizer *tokenizer) {
    اگر (tokenizer->init_ok) {
        برای (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            آزاد(tokenizer->token_table[i]);
        }
        آزاد(tokenizer->token_table);
    }
}
