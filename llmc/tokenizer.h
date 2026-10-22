/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
This is all we need for unconditional generation.
If we wanted to later prompt the model, we'd have to add decoding.
Which could be tricky in C because of the regex involved, to look into later.
*/

#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <ctype.h>
#অন্তর্ভাব <assert.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#অন্তর্ভাব "utils.h"

// ----------------------------------------------------------------------------

প্রকারনির্দেশ সংরচনা {
    uint32_t vocab_size;
    বর্ণ **token_table;
    পূর্ণাঙ্ক init_ok;
    পূর্ণাঙ্ক eot_token; // <|endoftext|> token id
} Tokenizer;

শূন্য safe_printf(ধ্রুব বর্ণ *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    যদি (piece == রিক্ত) { প্রত্যাগম; }
    যদি (piece[0] == '\0') { প্রত্যাগম; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    যদি (piece[1] == '\0') {
        unsigned বর্ণ byte_val = piece[0];
        যদি (!(isprint(byte_val) || isspace(byte_val))) {
            প্রত্যাগম; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}

শূন্য tokenizer_init(Tokenizer *tokenizer, ধ্রুব বর্ণ *filename) {
    FILE *file = fopen(filename, "rb");
    যদি (file == রিক্ত) {
        // try to be more helpful as we just added this feature, erase later
        printf("---\n");
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added April 14 2024.\n");
        printf("Re-run `python train_gpt2.py` to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        প্রত্যাগম;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, পরিমাণ(uint32_t), 256, file);
    প্রতিজ্ঞা(header[0] == 20240328);
    পূর্ণাঙ্ক version = header[1];
    tokenizer->vocab_size = header[2];
    যদি (version == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        প্রতিজ্ঞা(tokenizer->vocab_size == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } অন্যথা যদি (version == 2) {
        tokenizer->eot_token = header[3];
    } অন্যথা {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, version);
        exit(EXIT_FAILURE);
    }
    // read in all the tokens
    unsigned বর্ণ length;
    tokenizer->token_table = (বর্ণ **)mallocCheck(tokenizer->vocab_size * পরিমাণ(বর্ণ *));
    প্রতি (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        freadCheck(&length, পরিমাণ(unsigned বর্ণ), 1, file);
        প্রতিজ্ঞা(length > 0); // every token should be at least one character
        বর্ণ *token_bytes = (বর্ণ *)mallocCheck(length + 1);
        freadCheck(token_bytes, পরিমাণ(বর্ণ), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

ধ্রুব বর্ণ *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    যদি (tokenizer->init_ok == 0) {
        প্রত্যাগম রিক্ত;
    }
    যদি (token_id < tokenizer->vocab_size) {
        প্রত্যাগম tokenizer->token_table[token_id];
    } অন্যথা {
        printf("invalid token id %u!\n", token_id);
        প্রত্যাগম রিক্ত;
    }
}

শূন্য tokenizer_free(Tokenizer *tokenizer) {
    যদি (tokenizer->init_ok) {
        প্রতি (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            মুক্ত(tokenizer->token_table[i]);
        }
        মুক্ত(tokenizer->token_table);
    }
}
