/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
This is all we need for unconditional generation.
If we wanted to later prompt the model, we'd have to add decoding.
Which could be tricky in C because of the regex involved, to look into later.
*/

#అన్తర్భావ <stdint.h>
#అన్తర్భావ <ctype.h>
#అన్తర్భావ <assert.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#అన్తర్భావ "utils.h"

// ----------------------------------------------------------------------------

రకనిర్వచనం నిర్మాణం {
    uint32_t vocab_కొలత;
    అక్షరం **token_పట్టిక;
    పూర్ణాంకం init_ok;
    పూర్ణాంకం eot_token; // <|endoftext|> token id
} Tokenizer;

శూన్య సురక్షిత_printf(స్థిరరాశి అక్షరం *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    ఒకవేళ (piece == ఖాళీ) { తిరిగి; }
    ఒకవేళ (piece[0] == '\0') { తిరిగి; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    ఒకవేళ (piece[1] == '\0') {
        గుర్తులేని అక్షరం byte_val = piece[0];
        ఒకవేళ (!(isprint(byte_val) || ఖాళీనా(byte_val))) {
            తిరిగి; // weird byte, don't print it
        }
    }
    రాయి("%s", piece);
}

శూన్య tokenizer_init(Tokenizer *tokenizer, స్థిరరాశి అక్షరం *filename) {
    FILE *file = fopen(filename, "rb");
    ఒకవేళ (file == ఖాళీ) {
        // try to be more helpful as we just added this feature, erase later
        రాయి("---\n");
        రాయి("WARNING: Failed to open the tokenizer file %s\n", filename);
        రాయి("The Tokenizer is a new feature added April 14 2024.\n");
        రాయి("Re-run `python train_gpt2.py` to write it\n");
        రాయి("---\n");
        tokenizer->init_ok = 0;
        తిరిగి;
    }
    // read in the header
    uint32_t శీర్షిక[256];
    freadCheck(శీర్షిక, పరిమాణం(uint32_t), 256, file);
    ప్రతిజ్ఞా(శీర్షిక[0] == 20240328);
    పూర్ణాంకం సంస్కరణ = శీర్షిక[1];
    tokenizer->vocab_కొలత = శీర్షిక[2];
    ఒకవేళ (సంస్కరణ == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        ప్రతిజ్ఞా(tokenizer->vocab_కొలత == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } లేకపోతే ఒకవేళ (సంస్కరణ == 2) {
        tokenizer->eot_token = శీర్షిక[3];
    } లేకపోతే {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, సంస్కరణ);
        నిష్క్రమించు(EXIT_FAILURE);
    }
    // read in all the tokens
    గుర్తులేని అక్షరం నిడివి;
    tokenizer->token_పట్టిక = (అక్షరం **)mallocCheck(tokenizer->vocab_కొలత * పరిమాణం(అక్షరం *));
    కొరకు (uint32_t i = 0; i < tokenizer->vocab_కొలత; i++) {
        freadCheck(&నిడివి, పరిమాణం(గుర్తులేని అక్షరం), 1, file);
        ప్రతిజ్ఞా(నిడివి > 0); // every token should be at least one character
        అక్షరం *token_బైట్లు = (అక్షరం *)mallocCheck(నిడివి + 1);
        freadCheck(token_బైట్లు, పరిమాణం(అక్షరం), నిడివి, file);
        token_బైట్లు[నిడివి] = '\0';  // Add null terminator for printing
        tokenizer->token_పట్టిక[i] = token_బైట్లు;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

స్థిరరాశి అక్షరం *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    ఒకవేళ (tokenizer->init_ok == 0) {
        తిరిగి ఖాళీ;
    }
    ఒకవేళ (token_id < tokenizer->vocab_కొలత) {
        తిరిగి tokenizer->token_పట్టిక[token_id];
    } లేకపోతే {
        రాయి("invalid token id %u!\n", token_id);
        తిరిగి ఖాళీ;
    }
}

శూన్య tokenizer_free(Tokenizer *tokenizer) {
    ఒకవేళ (tokenizer->init_ok) {
        కొరకు (uint32_t i = 0; i < tokenizer->vocab_కొలత; i++) {
            విడుదల(tokenizer->token_పట్టిక[i]);
        }
        విడుదల(tokenizer->token_పట్టిక);
    }
}
