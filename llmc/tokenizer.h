/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
This is all we need for unconditional generation.
If we wanted to later prompt the model, we'd have to add decoding.
Which could be tricky in C because of the regex involved, to look into later.
*/

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <ctype.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#ਅਨ੍ਤਰ੍ਭਾਵ "utils.h"

// ----------------------------------------------------------------------------

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    uint32_t vocab_ਮਾਪ;
    ਅੱਖਰ **token_ਸਾਰਣੀ;
    ਪੂਰਨ init_ok;
    ਪੂਰਨ eot_token; // <|endoftext|> token id
} Tokenizer;

ਖਾਲੀ ਸੁਰੱਖਿਅਤ_printf(ਅਚਲ ਅੱਖਰ *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    ਜੇ (piece == ਨਿਰਾ) { ਵਾਪਸ; }
    ਜੇ (piece[0] == '\0') { ਵਾਪਸ; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    ਜੇ (piece[1] == '\0') {
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ byte_val = piece[0];
        ਜੇ (!(isprint(byte_val) || ਖਾਲੀ_ਥਾਂ_ਹੈ(byte_val))) {
            ਵਾਪਸ; // weird byte, don't print it
        }
    }
    ਲਿਖੋ("%s", piece);
}

ਖਾਲੀ tokenizer_init(Tokenizer *tokenizer, ਅਚਲ ਅੱਖਰ *filename) {
    FILE *file = fopen(filename, "rb");
    ਜੇ (file == ਨਿਰਾ) {
        // try to be more helpful as we just added this feature, erase later
        ਲਿਖੋ("---\n");
        ਲਿਖੋ("WARNING: Failed to open the tokenizer file %s\n", filename);
        ਲਿਖੋ("The Tokenizer is a new feature added April 14 2024.\n");
        ਲਿਖੋ("Re-run `python train_gpt2.py` to write it\n");
        ਲਿਖੋ("---\n");
        tokenizer->init_ok = 0;
        ਵਾਪਸ;
    }
    // read in the header
    uint32_t ਸਿਰਲੇਖ[256];
    freadCheck(ਸਿਰਲੇਖ, ਆਕਾਰ(uint32_t), 256, file);
    ਪ੍ਰਤਿਜ੍ਞਾ(ਸਿਰਲੇਖ[0] == 20240328);
    ਪੂਰਨ ਸੰਸਕਰਣ = ਸਿਰਲੇਖ[1];
    tokenizer->vocab_ਮਾਪ = ਸਿਰਲੇਖ[2];
    ਜੇ (ਸੰਸਕਰਣ == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        ਪ੍ਰਤਿਜ੍ਞਾ(tokenizer->vocab_ਮਾਪ == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } ਨਹੀਂਤਾਂ ਜੇ (ਸੰਸਕਰਣ == 2) {
        tokenizer->eot_token = ਸਿਰਲੇਖ[3];
    } ਨਹੀਂਤਾਂ {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, ਸੰਸਕਰਣ);
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
    // read in all the tokens
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ ਲੰਮਾਈ;
    tokenizer->token_ਸਾਰਣੀ = (ਅੱਖਰ **)mallocCheck(tokenizer->vocab_ਮਾਪ * ਆਕਾਰ(ਅੱਖਰ *));
    ਲਈ (uint32_t i = 0; i < tokenizer->vocab_ਮਾਪ; i++) {
        freadCheck(&ਲੰਮਾਈ, ਆਕਾਰ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ), 1, file);
        ਪ੍ਰਤਿਜ੍ਞਾ(ਲੰਮਾਈ > 0); // every token should be at least one character
        ਅੱਖਰ *token_ਬਾਈਟਾਂ = (ਅੱਖਰ *)mallocCheck(ਲੰਮਾਈ + 1);
        freadCheck(token_ਬਾਈਟਾਂ, ਆਕਾਰ(ਅੱਖਰ), ਲੰਮਾਈ, file);
        token_ਬਾਈਟਾਂ[ਲੰਮਾਈ] = '\0';  // Add null terminator for printing
        tokenizer->token_ਸਾਰਣੀ[i] = token_ਬਾਈਟਾਂ;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

ਅਚਲ ਅੱਖਰ *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    ਜੇ (tokenizer->init_ok == 0) {
        ਵਾਪਸ ਨਿਰਾ;
    }
    ਜੇ (token_id < tokenizer->vocab_ਮਾਪ) {
        ਵਾਪਸ tokenizer->token_ਸਾਰਣੀ[token_id];
    } ਨਹੀਂਤਾਂ {
        ਲਿਖੋ("invalid token id %u!\n", token_id);
        ਵਾਪਸ ਨਿਰਾ;
    }
}

ਖਾਲੀ tokenizer_free(Tokenizer *tokenizer) {
    ਜੇ (tokenizer->init_ok) {
        ਲਈ (uint32_t i = 0; i < tokenizer->vocab_ਮਾਪ; i++) {
            ਮੁਕਤ(tokenizer->token_ਸਾਰਣੀ[i]);
        }
        ਮੁਕਤ(tokenizer->token_ਸਾਰਣੀ);
    }
}
