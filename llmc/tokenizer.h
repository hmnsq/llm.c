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

قسمتعریف ساخت {
    uint32_t vocab_سائز;
    حرف **token_جدول;
    صحیح init_ok;
    صحیح eot_token; // <|endoftext|> token id
} Tokenizer;

خالی محفوظ_printf(ثابت حرف *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    اگر (piece == تہی) { واپس; }
    اگر (piece[0] == '\0') { واپس; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    اگر (piece[1] == '\0') {
        غیرعلامتی حرف byte_val = piece[0];
        اگر (!(isprint(byte_val) || خلا_ہے(byte_val))) {
            واپس; // weird byte, don't print it
        }
    }
    لکھو("%s", piece);
}

خالی tokenizer_init(Tokenizer *tokenizer, ثابت حرف *filename) {
    FILE *file = fopen(filename, "rb");
    اگر (file == تہی) {
        // try to be more helpful as we just added this feature, erase later
        لکھو("---\n");
        لکھو("WARNING: Failed to open the tokenizer file %s\n", filename);
        لکھو("The Tokenizer is a new feature added April 14 2024.\n");
        لکھو("Re-run `python train_gpt2.py` to write it\n");
        لکھو("---\n");
        tokenizer->init_ok = 0;
        واپس;
    }
    // read in the header
    uint32_t سرخی[256];
    freadCheck(سرخی, حجم(uint32_t), 256, file);
    أكد(سرخی[0] == 20240328);
    صحیح نسخہ = سرخی[1];
    tokenizer->vocab_سائز = سرخی[2];
    اگر (نسخہ == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        أكد(tokenizer->vocab_سائز == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } ورنہ اگر (نسخہ == 2) {
        tokenizer->eot_token = سرخی[3];
    } ورنہ {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, نسخہ);
        باہر_نکلو(EXIT_FAILURE);
    }
    // read in all the tokens
    غیرعلامتی حرف لمبائی_2;
    tokenizer->token_جدول = (حرف **)mallocCheck(tokenizer->vocab_سائز * حجم(حرف *));
    برائے (uint32_t i = 0; i < tokenizer->vocab_سائز; i++) {
        freadCheck(&لمبائی_2, حجم(غیرعلامتی حرف), 1, file);
        أكد(لمبائی_2 > 0); // every token should be at least one character
        حرف *token_بائٹیں = (حرف *)mallocCheck(لمبائی_2 + 1);
        freadCheck(token_بائٹیں, حجم(حرف), لمبائی_2, file);
        token_بائٹیں[لمبائی_2] = '\0';  // Add null terminator for printing
        tokenizer->token_جدول[i] = token_بائٹیں;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

ثابت حرف *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    اگر (tokenizer->init_ok == 0) {
        واپس تہی;
    }
    اگر (token_id < tokenizer->vocab_سائز) {
        واپس tokenizer->token_جدول[token_id];
    } ورنہ {
        لکھو("invalid token id %u!\n", token_id);
        واپس تہی;
    }
}

خالی tokenizer_free(Tokenizer *tokenizer) {
    اگر (tokenizer->init_ok) {
        برائے (uint32_t i = 0; i < tokenizer->vocab_سائز; i++) {
            آزاد(tokenizer->token_جدول[i]);
        }
        آزاد(tokenizer->token_جدول);
    }
}
