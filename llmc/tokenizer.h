/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
This is all we need for unconditional generation.
If we wanted to later prompt the model, we'd have to add decoding.
Which could be tricky in C because of the regex involved, to look into later.
*/

#समावेश <stdint.h>
#समावेश <ctype.h>
#समावेश <assert.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#समावेश "utils.h"

// ----------------------------------------------------------------------------

प्रकार काष्ठा {
    uint32_t vocab_आकार;
    अक्षर **token_table;
    पूर्णांक init_ok;
    पूर्णांक eot_token; // <|endoftext|> token id
} Tokenizer;

व्योम safe_printf(स्थिर अक्षर *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    अगर (piece == शून्य) { वापस; }
    अगर (piece[0] == '\0') { वापस; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    अगर (piece[1] == '\0') {
        अचिन्हित अक्षर byte_val = piece[0];
        अगर (!(है_छाप(byte_val) || है_खाली(byte_val))) {
            वापस; // weird byte, don't print it
        }
    }
    म_लिखो("%s", piece);
}

व्योम tokenizer_init(Tokenizer *tokenizer, स्थिर अक्षर *filename) {
    खाता *file = ख_खोलो(filename, "rb");
    अगर (file == शून्य) {
        // try to be more helpful as we just added this feature, erase later
        म_लिखो("---\n");
        म_लिखो("WARNING: Failed to open the tokenizer file %s\n", filename);
        म_लिखो("The Tokenizer is a new feature added April 14 2024.\n");
        म_लिखो("Re-run `python train_gpt2.py` to write it\n");
        म_लिखो("---\n");
        tokenizer->init_ok = 0;
        वापस;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, माप(uint32_t), 256, file);
    निश्चित(header[0] == 20240328);
    पूर्णांक version = header[1];
    tokenizer->vocab_आकार = header[2];
    अगर (version == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        निश्चित(tokenizer->vocab_आकार == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } अन्यथा अगर (version == 2) {
        tokenizer->eot_token = header[3];
    } अन्यथा {
        ख_लिखो(मानक_त्रुटि, "Tokenizer model file %s has bad version: %d\n", filename, version);
        निकास(निकास_त्रुटि);
    }
    // read in all the tokens
    अचिन्हित अक्षर लंबाई;
    tokenizer->token_table = (अक्षर **)mallocCheck(tokenizer->vocab_आकार * माप(अक्षर *));
    क्रम (uint32_t i = 0; i < tokenizer->vocab_आकार; i++) {
        freadCheck(&लंबाई, माप(अचिन्हित अक्षर), 1, file);
        निश्चित(लंबाई > 0); // every token should be at least one character
        अक्षर *token_bytes = (अक्षर *)mallocCheck(लंबाई + 1);
        freadCheck(token_bytes, माप(अक्षर), लंबाई, file);
        token_bytes[लंबाई] = '\0';  // Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

स्थिर अक्षर *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    अगर (tokenizer->init_ok == 0) {
        वापस शून्य;
    }
    अगर (token_id < tokenizer->vocab_आकार) {
        वापस tokenizer->token_table[token_id];
    } अन्यथा {
        म_लिखो("invalid token id %u!\n", token_id);
        वापस शून्य;
    }
}

व्योम tokenizer_free(Tokenizer *tokenizer) {
    अगर (tokenizer->init_ok) {
        क्रम (uint32_t i = 0; i < tokenizer->vocab_आकार; i++) {
            मुक्त(tokenizer->token_table[i]);
        }
        मुक्त(tokenizer->token_table);
    }
}
