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
    uint32_t vocab_سائيز;
    حرف **token_جدول;
    صحيح init_ok;
    صحيح eot_token; // <|endoftext|> token id
} Tokenizer;

فراغ محفوظ_printf(ثابت حرف *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    إذا (piece == عدم) { أرجع; }
    إذا (piece[0] == '\0') { أرجع; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    إذا (piece[1] == '\0') {
        بنانشان حرف byte_val = piece[0];
        إذا (!(isprint(byte_val) || خال_آهي(byte_val))) {
            أرجع; // weird byte, don't print it
        }
    }
    لکو("%s", piece);
}

فراغ tokenizer_init(Tokenizer *tokenizer, ثابت حرف *filename) {
    FILE *file = fopen(filename, "rb");
    إذا (file == عدم) {
        // try to be more helpful as we just added this feature, erase later
        لکو("---\n");
        لکو("WARNING: Failed to open the tokenizer file %s\n", filename);
        لکو("The Tokenizer is a new feature added April 14 2024.\n");
        لکو("Re-run `python train_gpt2.py` to write it\n");
        لکو("---\n");
        tokenizer->init_ok = 0;
        أرجع;
    }
    // read in the header
    uint32_t سرو[256];
    freadCheck(سرو, حجم(uint32_t), 256, file);
    أكد(سرو[0] == 20240328);
    صحيح نسخو = سرو[1];
    tokenizer->vocab_سائيز = سرو[2];
    إذا (نسخو == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        أكد(tokenizer->vocab_سائيز == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } وإلا إذا (نسخو == 2) {
        tokenizer->eot_token = سرو[3];
    } وإلا {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, نسخو);
        ٻاهر_نڪرو(EXIT_FAILURE);
    }
    // read in all the tokens
    بنانشان حرف ڊيگهائي;
    tokenizer->token_جدول = (حرف **)mallocCheck(tokenizer->vocab_سائيز * حجم(حرف *));
    لكل (uint32_t i = 0; i < tokenizer->vocab_سائيز; i++) {
        freadCheck(&ڊيگهائي, حجم(بنانشان حرف), 1, file);
        أكد(ڊيگهائي > 0); // every token should be at least one character
        حرف *token_بائيٽون = (حرف *)mallocCheck(ڊيگهائي + 1);
        freadCheck(token_بائيٽون, حجم(حرف), ڊيگهائي, file);
        token_بائيٽون[ڊيگهائي] = '\0';  // Add null terminator for printing
        tokenizer->token_جدول[i] = token_بائيٽون;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

ثابت حرف *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    إذا (tokenizer->init_ok == 0) {
        أرجع عدم;
    }
    إذا (token_id < tokenizer->vocab_سائيز) {
        أرجع tokenizer->token_جدول[token_id];
    } وإلا {
        لکو("invalid token id %u!\n", token_id);
        أرجع عدم;
    }
}

فراغ tokenizer_free(Tokenizer *tokenizer) {
    إذا (tokenizer->init_ok) {
        لكل (uint32_t i = 0; i < tokenizer->vocab_سائيز; i++) {
            حرر(tokenizer->token_جدول[i]);
        }
        حرر(tokenizer->token_جدول);
    }
}
