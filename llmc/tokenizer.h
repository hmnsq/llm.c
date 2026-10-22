/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
This is all we need for unconditional generation.
If we wanted to later prompt the model, we'd have to add decoding.
Which could be tricky in C because of the regex involved, to look into later.
*/

#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <ctype.h>
#અન્તર્ભાવ <assert.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#અન્તર્ભાવ "utils.h"

// ----------------------------------------------------------------------------

પ્રકારવ્યાખ્યા સંરચના {
    uint32_t vocab_size;
    અક્ષર **token_table;
    પૂર્ણાંક init_ok;
    પૂર્ણાંક eot_token; // <|endoftext|> token id
} Tokenizer;

શૂન્ય safe_printf(અચલ અક્ષર *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    જો (piece == ખાલી) { પાછા; }
    જો (piece[0] == '\0') { પાછા; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    જો (piece[1] == '\0') {
        unsigned અક્ષર byte_val = piece[0];
        જો (!(isprint(byte_val) || isspace(byte_val))) {
            પાછા; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}

શૂન્ય tokenizer_init(Tokenizer *tokenizer, અચલ અક્ષર *filename) {
    FILE *file = fopen(filename, "rb");
    જો (file == ખાલી) {
        // try to be more helpful as we just added this feature, erase later
        printf("---\n");
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added April 14 2024.\n");
        printf("Re-run `python train_gpt2.py` to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        પાછા;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, કદ(uint32_t), 256, file);
    પ્રતિજ્ઞા(header[0] == 20240328);
    પૂર્ણાંક version = header[1];
    tokenizer->vocab_size = header[2];
    જો (version == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        પ્રતિજ્ઞા(tokenizer->vocab_size == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } નહીંતર જો (version == 2) {
        tokenizer->eot_token = header[3];
    } નહીંતર {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, version);
        exit(EXIT_FAILURE);
    }
    // read in all the tokens
    unsigned અક્ષર length;
    tokenizer->token_table = (અક્ષર **)mallocCheck(tokenizer->vocab_size * કદ(અક્ષર *));
    માટે (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        freadCheck(&length, કદ(unsigned અક્ષર), 1, file);
        પ્રતિજ્ઞા(length > 0); // every token should be at least one character
        અક્ષર *token_bytes = (અક્ષર *)mallocCheck(length + 1);
        freadCheck(token_bytes, કદ(અક્ષર), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

અચલ અક્ષર *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    જો (tokenizer->init_ok == 0) {
        પાછા ખાલી;
    }
    જો (token_id < tokenizer->vocab_size) {
        પાછા tokenizer->token_table[token_id];
    } નહીંતર {
        printf("invalid token id %u!\n", token_id);
        પાછા ખાલી;
    }
}

શૂન્ય tokenizer_free(Tokenizer *tokenizer) {
    જો (tokenizer->init_ok) {
        માટે (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            મુક્ત(tokenizer->token_table[i]);
        }
        મુક્ત(tokenizer->token_table);
    }
}
