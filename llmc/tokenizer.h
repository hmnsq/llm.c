/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
This is all we need for unconditional generation.
If we wanted to later prompt the model, we'd have to add decoding.
Which could be tricky in C because of the regex involved, to look into later.
*/

#अन्तर्भाव <stdint.h>
#अन्तर्भाव <ctype.h>
#अन्तर्भाव <assert.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#अन्तर्भाव "utils.h"

// ----------------------------------------------------------------------------

प्रकारनिर्देश संरचना {
    uint32_t vocab_size;
    वर्ण **token_table;
    पूर्णाङ्क init_ok;
    पूर्णाङ्क eot_token; // <|endoftext|> token id
} Tokenizer;

शून्य safe_printf(अचर वर्ण *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    यदि (piece == रिक्त) { फर्काऊ; }
    यदि (piece[0] == '\0') { फर्काऊ; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    यदि (piece[1] == '\0') {
        अचिन्हित वर्ण byte_val = piece[0];
        यदि (!(है_छाप(byte_val) || है_खाली(byte_val))) {
            फर्काऊ; // weird byte, don't print it
        }
    }
    म_लिखो("%s", piece);
}

शून्य tokenizer_init(Tokenizer *tokenizer, अचर वर्ण *filename) {
    खाता *file = ख_खोलो(filename, "rb");
    यदि (file == रिक्त) {
        // try to be more helpful as we just added this feature, erase later
        म_लिखो("---\n");
        म_लिखो("WARNING: Failed to open the tokenizer file %s\n", filename);
        म_लिखो("The Tokenizer is a new feature added April 14 2024.\n");
        म_लिखो("Re-run `python train_gpt2.py` to write it\n");
        म_लिखो("---\n");
        tokenizer->init_ok = 0;
        फर्काऊ;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, परिमाण(uint32_t), 256, file);
    प्रतिज्ञा(header[0] == 20240328);
    पूर्णाङ्क version = header[1];
    tokenizer->vocab_size = header[2];
    यदि (version == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        प्रतिज्ञा(tokenizer->vocab_size == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } अन्यथा यदि (version == 2) {
        tokenizer->eot_token = header[3];
    } अन्यथा {
        ख_लिखो(मानक_त्रुटि, "Tokenizer model file %s has bad version: %d\n", filename, version);
        निकास(निकास_त्रुटि);
    }
    // read in all the tokens
    अचिन्हित वर्ण length;
    tokenizer->token_table = (वर्ण **)mallocCheck(tokenizer->vocab_size * परिमाण(वर्ण *));
    लागि (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        freadCheck(&length, परिमाण(अचिन्हित वर्ण), 1, file);
        प्रतिज्ञा(length > 0); // every token should be at least one character
        वर्ण *token_bytes = (वर्ण *)mallocCheck(length + 1);
        freadCheck(token_bytes, परिमाण(वर्ण), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

अचर वर्ण *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    यदि (tokenizer->init_ok == 0) {
        फर्काऊ रिक्त;
    }
    यदि (token_id < tokenizer->vocab_size) {
        फर्काऊ tokenizer->token_table[token_id];
    } अन्यथा {
        म_लिखो("invalid token id %u!\n", token_id);
        फर्काऊ रिक्त;
    }
}

शून्य tokenizer_free(Tokenizer *tokenizer) {
    यदि (tokenizer->init_ok) {
        लागि (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            मुक्त(tokenizer->token_table[i]);
        }
        मुक्त(tokenizer->token_table);
    }
}
