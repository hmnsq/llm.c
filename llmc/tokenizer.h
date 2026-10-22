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
    uint32_t vocab_اندازه;
    حرف **token_جدول;
    صحيح init_ok;
    صحيح eot_token; // <|endoftext|> token id
} Tokenizer;

فراغ امن_printf(ثابت حرف *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    إذا (piece == عدم) { أرجع; }
    إذا (piece[0] == '\0') { أرجع; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    إذا (piece[1] == '\0') {
        بیعلامت حرف byte_val = piece[0];
        إذا (!(isprint(byte_val) || آیا_فاصله(byte_val))) {
            أرجع; // weird byte, don't print it
        }
    }
    چاپ_کن("%s", piece);
}

فراغ tokenizer_init(Tokenizer *tokenizer, ثابت حرف *filename) {
    FILE *file = fopen(filename, "rb");
    إذا (file == عدم) {
        // try to be more helpful as we just added this feature, erase later
        چاپ_کن("---\n");
        چاپ_کن("WARNING: Failed to open the tokenizer file %s\n", filename);
        چاپ_کن("The Tokenizer is a new feature added April 14 2024.\n");
        چاپ_کن("Re-run `python train_gpt2.py` to write it\n");
        چاپ_کن("---\n");
        tokenizer->init_ok = 0;
        أرجع;
    }
    // read in the header
    uint32_t سرآیند[256];
    freadCheck(سرآیند, حجم(uint32_t), 256, file);
    أكد(سرآیند[0] == 20240328);
    صحيح نسخه = سرآیند[1];
    tokenizer->vocab_اندازه = سرآیند[2];
    إذا (نسخه == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        أكد(tokenizer->vocab_اندازه == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } وإلا إذا (نسخه == 2) {
        tokenizer->eot_token = سرآیند[3];
    } وإلا {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, نسخه);
        خروج(EXIT_FAILURE);
    }
    // read in all the tokens
    بیعلامت حرف درازی;
    tokenizer->token_جدول = (حرف **)mallocCheck(tokenizer->vocab_اندازه * حجم(حرف *));
    لكل (uint32_t i = 0; i < tokenizer->vocab_اندازه; i++) {
        freadCheck(&درازی, حجم(بیعلامت حرف), 1, file);
        أكد(درازی > 0); // every token should be at least one character
        حرف *token_بایتها = (حرف *)mallocCheck(درازی + 1);
        freadCheck(token_بایتها, حجم(حرف), درازی, file);
        token_بایتها[درازی] = '\0';  // Add null terminator for printing
        tokenizer->token_جدول[i] = token_بایتها;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

ثابت حرف *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    إذا (tokenizer->init_ok == 0) {
        أرجع عدم;
    }
    إذا (token_id < tokenizer->vocab_اندازه) {
        أرجع tokenizer->token_جدول[token_id];
    } وإلا {
        چاپ_کن("invalid token id %u!\n", token_id);
        أرجع عدم;
    }
}

فراغ tokenizer_free(Tokenizer *tokenizer) {
    إذا (tokenizer->init_ok) {
        لكل (uint32_t i = 0; i < tokenizer->vocab_اندازه; i++) {
            حرر(tokenizer->token_جدول[i]);
        }
        حرر(tokenizer->token_جدول);
    }
}
