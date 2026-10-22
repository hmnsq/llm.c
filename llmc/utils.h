/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#ifndef UTILS_H
#define UTILS_H

#ଅନ୍ତର୍ଭାଵ <unistd.h>
#ଅନ୍ତର୍ଭାଵ <string.h>
#ଅନ୍ତର୍ଭାଵ <stdio.h>
#ଅନ୍ତର୍ଭାଵ <stdlib.h>
#ଅନ୍ତର୍ଭାଵ <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#ifndef _WIN32
#ଅନ୍ତର୍ଭାଵ <dirent.h>
#ଅନ୍ତର୍ଭାଵ <arpa/inet.h>
#endif

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

ବାହ୍ୟ ଧାଡ଼ିଭିତରେ FILE *fopen_check(ଅଚଳ ଅକ୍ଷର *path, ଅଚଳ ଅକ୍ଷର *mode, ଅଚଳ ଅକ୍ଷର *file, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଧାଡ଼ି) {
    FILE *fp = fopen(path, mode);
    ଯଦି (fp == ଖାଲି) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, ଧାଡ଼ି);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ଧାଡ଼ି);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    ଫେରନ୍ତୁ fp;
}

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

ବାହ୍ୟ ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ fread_check(ଶୂନ୍ୟ *ptr, size_t size, size_t nmemb, FILE *stream, ଅଚଳ ଅକ୍ଷର *file, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଧାଡ଼ି) {
    size_t result = fread(ptr, size, nmemb, stream);
    ଯଦି (result != nmemb) {
        ଯଦି (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, ଧାଡ଼ି);
        } ଅନ୍ୟଥା ଯଦି (ferror(stream)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, ଧାଡ଼ି);
        } ଅନ୍ୟଥା {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, ଧାଡ଼ି, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ଧାଡ଼ି);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

ବାହ୍ୟ ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ fclose_check(FILE *fp, ଅଚଳ ଅକ୍ଷର *file, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଧାଡ଼ି) {
    ଯଦି (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, ଧାଡ଼ି);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ଧାଡ଼ି);
        exit(EXIT_FAILURE);
    }
}

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

ବାହ୍ୟ ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ sclose_check(ପୂର୍ଣ୍ଣସଂଖ୍ୟା sockfd, ଅଚଳ ଅକ୍ଷର *file, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଧାଡ଼ି) {
    ଯଦି (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, ଧାଡ଼ି);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ଧାଡ଼ି);
        exit(EXIT_FAILURE);
    }
}

#define scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#ifdef _WIN32
ବାହ୍ୟ ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ closesocket_check(ପୂର୍ଣ୍ଣସଂଖ୍ୟା sockfd, ଅଚଳ ଅକ୍ଷର *file, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଧାଡ଼ି) {
    ଯଦି (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, ଧାଡ଼ି);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ଧାଡ଼ି);
        exit(EXIT_FAILURE);
    }
}

#define closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#endif

ବାହ୍ୟ ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ fseek_check(FILE *fp, long off, ପୂର୍ଣ୍ଣସଂଖ୍ୟା whence, ଅଚଳ ଅକ୍ଷର *file, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଧାଡ଼ି) {
    ଯଦି (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, ଧାଡ଼ି);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", ଧାଡ଼ି);
        exit(EXIT_FAILURE);
    }
}

#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

ବାହ୍ୟ ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ fwrite_check(ଶୂନ୍ୟ *ptr, size_t size, size_t nmemb, FILE *stream, ଅଚଳ ଅକ୍ଷର *file, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଧାଡ଼ି) {
    size_t result = fwrite(ptr, size, nmemb, stream);
    ଯଦି (result != nmemb) {
        ଯଦି (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, ଧାଡ଼ି);
        } ଅନ୍ୟଥା ଯଦି (ferror(stream)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, ଧାଡ଼ି);
        } ଅନ୍ୟଥା {
            fprintf(stderr, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, ଧାଡ଼ି, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ଧାଡ଼ି);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Written elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define fwriteCheck(ptr, size, nmemb, stream) fwrite_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

ବାହ୍ୟ ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ *malloc_check(size_t size, ଅଚଳ ଅକ୍ଷର *file, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଧାଡ଼ି) {
    ଶୂନ୍ୟ *ptr = malloc(size);
    ଯଦି (ptr == ଖାଲି) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, ଧାଡ଼ି);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ଧାଡ଼ି);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    ଫେରନ୍ତୁ ptr;
}

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
ବାହ୍ୟ ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ token_check(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା* tokens, ପୂର୍ଣ୍ଣସଂଖ୍ୟା token_ଗଣନା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vocab_size, ଅଚଳ ଅକ୍ଷର *file, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଧାଡ଼ି) {
    ପାଇଁ(ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < token_ଗଣନା; i++) {
        ଯଦି(!(0 <= tokens[i] && tokens[i] < vocab_size)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, ଧାଡ଼ି);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", ଧାଡ଼ି);
            fprintf(stderr, "  Token: %d\n", tokens[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_size);
            exit(EXIT_FAILURE);
        }
    }
}
#define tokenCheck(tokens, ଗଣନା_2, vocab) token_check(tokens, ଗଣନା_2, vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

ବାହ୍ୟ ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ create_dir_if_not_exists(ଅଚଳ ଅକ୍ଷର *dir) {
    ଯଦି (dir == ଖାଲି) { ଫେରନ୍ତୁ; }
    ସଂରଚନା stat st = {0};
    ଯଦି (stat(dir, &st) == -1) {
        ଯଦି (mkdir(dir, 0700) == -1) {
            printf("ERROR: could not create directory: %s\n", dir);
            exit(EXIT_FAILURE);
        }
        printf("created directory: %s\n", dir);
    }
}

ବାହ୍ୟ ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା find_max_step(ଅଚଳ ଅକ୍ଷର* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    ଯଦି (output_log_dir == ଖାଲି) { ଫେରନ୍ତୁ -1; }
    DIR* dir;
    ସଂରଚନା dirent* entry;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_step = -1;
    dir = opendir(output_log_dir);
    ଯଦି (dir == ଖାଲି) { ଫେରନ୍ତୁ -1; }
    ଯେତେବେଳେ ((entry = readdir(dir)) != ଖାଲି) {
        ଯଦି (strncmp(entry->d_name, "DONE_", 5) == 0) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା step = atoi(entry->d_name + 5);
            ଯଦି (step > max_step) {
                max_step = step;
            }
        }
    }
    closedir(dir);
    ଫେରନ୍ତୁ max_step;
}

ବାହ୍ୟ ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ends_with_bin(ଅଚଳ ଅକ୍ଷର* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    ଯଦି (str == ଖାଲି) { ଫେରନ୍ତୁ 0; }
    size_t len = strlen(str);
    ଅଚଳ ଅକ୍ଷର* suffix = ".bin";
    size_t suffix_len = strlen(suffix);
    ଯଦି (len < suffix_len) { ଫେରନ୍ତୁ 0; }
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା suffix_matches = strncmp(str + len - suffix_len, suffix, suffix_len) == 0;
    ଫେରନ୍ତୁ suffix_matches;
}

#endif