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

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    uint32_t vocab_ಗಾತ್ರ;
    ಅಕ್ಷರ **token_ಕೋಷ್ಟಕ;
    ಪೂರ್ಣಾಂಕ init_ok;
    ಪೂರ್ಣಾಂಕ eot_token; // <|endoftext|> token id
} Tokenizer;

ಶೂನ್ಯ ಸುರಕ್ಷಿತ_printf(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    ಆದರೆ (piece == ಖಾಲಿ) { ಹಿಂತಿರುಗಿ; }
    ಆದರೆ (piece[0] == '\0') { ಹಿಂತಿರುಗಿ; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    ಆದರೆ (piece[1] == '\0') {
        ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ byte_val = piece[0];
        ಆದರೆ (!(isprint(byte_val) || ಖಾಲಿಯೇ(byte_val))) {
            ಹಿಂತಿರುಗಿ; // weird byte, don't print it
        }
    }
    ಬರೆ("%s", piece);
}

ಶೂನ್ಯ tokenizer_init(Tokenizer *tokenizer, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *filename) {
    FILE *file = fopen(filename, "rb");
    ಆದರೆ (file == ಖಾಲಿ) {
        // try to be more helpful as we just added this feature, erase later
        ಬರೆ("---\n");
        ಬರೆ("WARNING: Failed to open the tokenizer file %s\n", filename);
        ಬರೆ("The Tokenizer is a new feature added April 14 2024.\n");
        ಬರೆ("Re-run `python train_gpt2.py` to write it\n");
        ಬರೆ("---\n");
        tokenizer->init_ok = 0;
        ಹಿಂತಿರುಗಿ;
    }
    // read in the header
    uint32_t ಶೀರ್ಷಿಕೆ[256];
    freadCheck(ಶೀರ್ಷಿಕೆ, ಗಾತ್ರ(uint32_t), 256, file);
    ಪ್ರತಿಜ್ಞಾ(ಶೀರ್ಷಿಕೆ[0] == 20240328);
    ಪೂರ್ಣಾಂಕ ಆವೃತ್ತಿ = ಶೀರ್ಷಿಕೆ[1];
    tokenizer->vocab_ಗಾತ್ರ = ಶೀರ್ಷಿಕೆ[2];
    ಆದರೆ (ಆವೃತ್ತಿ == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        ಪ್ರತಿಜ್ಞಾ(tokenizer->vocab_ಗಾತ್ರ == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಆವೃತ್ತಿ == 2) {
        tokenizer->eot_token = ಶೀರ್ಷಿಕೆ[3];
    } ಇಲ್ಲದಿದ್ದರೆ {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, ಆವೃತ್ತಿ);
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
    // read in all the tokens
    ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ ಉದ್ದಳತೆ;
    tokenizer->token_ಕೋಷ್ಟಕ = (ಅಕ್ಷರ **)mallocCheck(tokenizer->vocab_ಗಾತ್ರ * ಗಾತ್ರ(ಅಕ್ಷರ *));
    ಗಾಗಿ (uint32_t i = 0; i < tokenizer->vocab_ಗಾತ್ರ; i++) {
        freadCheck(&ಉದ್ದಳತೆ, ಗಾತ್ರ(ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ), 1, file);
        ಪ್ರತಿಜ್ಞಾ(ಉದ್ದಳತೆ > 0); // every token should be at least one character
        ಅಕ್ಷರ *token_ಬೈಟುಗಳು = (ಅಕ್ಷರ *)mallocCheck(ಉದ್ದಳತೆ + 1);
        freadCheck(token_ಬೈಟುಗಳು, ಗಾತ್ರ(ಅಕ್ಷರ), ಉದ್ದಳತೆ, file);
        token_ಬೈಟುಗಳು[ಉದ್ದಳತೆ] = '\0';  // Add null terminator for printing
        tokenizer->token_ಕೋಷ್ಟಕ[i] = token_ಬೈಟುಗಳು;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    ಆದರೆ (tokenizer->init_ok == 0) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }
    ಆದರೆ (token_id < tokenizer->vocab_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ tokenizer->token_ಕೋಷ್ಟಕ[token_id];
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಬರೆ("invalid token id %u!\n", token_id);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }
}

ಶೂನ್ಯ tokenizer_free(Tokenizer *tokenizer) {
    ಆದರೆ (tokenizer->init_ok) {
        ಗಾಗಿ (uint32_t i = 0; i < tokenizer->vocab_ಗಾತ್ರ; i++) {
            ಬಿಡುಗಡೆ(tokenizer->token_ಕೋಷ್ಟಕ[i]);
        }
        ಬಿಡುಗಡೆ(tokenizer->token_ಕೋಷ್ಟಕ);
    }
}
