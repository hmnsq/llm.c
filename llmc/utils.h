/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#ifndef UTILS_H
#define UTILS_H

#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#ifndef _WIN32
#include <dirent.h>
#include <arpa/inet.h>
#endif

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

extern anta_hstha FILE *fopen_check(const var_na *path, const var_na *mode, const var_na *file, ཧྲིལ line) {
    FILE *fp = fopen(path, mode);
    གལ་སྲིད (fp == NULL) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    ལོག fp;
}

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

extern anta_hstha void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const var_na *file, ཧྲིལ line) {
    size_t result = fread(ptr, size, nmemb, stream);
    གལ་སྲིད (result != nmemb) {
        གལ་སྲིད (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, line);
        } དེ་མིན གལ་སྲིད (ferror(stream)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, line);
        } དེ་མིན {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, line, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

extern anta_hstha void fclose_check(FILE *fp, const var_na *file, ཧྲིལ line) {
    གལ་སྲིད (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

extern anta_hstha void sclose_check(ཧྲིལ sockfd, const var_na *file, ཧྲིལ line) {
    གལ་སྲིད (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#ifdef _WIN32
extern anta_hstha void closesocket_check(ཧྲིལ sockfd, const var_na *file, ཧྲིལ line) {
    གལ་སྲིད (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#endif

extern anta_hstha void fseek_check(FILE *fp, long off, ཧྲིལ whence, const var_na *file, ཧྲིལ line) {
    གལ་སྲིད (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

extern anta_hstha void fwrite_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const var_na *file, ཧྲིལ line) {
    size_t result = fwrite(ptr, size, nmemb, stream);
    གལ་སྲིད (result != nmemb) {
        གལ་སྲིད (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, line);
        } དེ་མིན གལ་སྲིད (ferror(stream)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, line);
        } དེ་མིན {
            fprintf(stderr, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, line, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Written elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define fwriteCheck(ptr, size, nmemb, stream) fwrite_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

extern anta_hstha void *malloc_check(size_t size, const var_na *file, ཧྲིལ line) {
    void *ptr = malloc(size);
    གལ་སྲིད (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    ལོག ptr;
}

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
extern anta_hstha void token_check(const ཧྲིལ* tokens, ཧྲིལ token_count, ཧྲིལ vocab_size, const var_na *file, ཧྲིལ line) {
    དོན་ལུ(ཧྲིལ i = 0; i < token_count; i++) {
        གལ་སྲིད(!(0 <= tokens[i] && tokens[i] < vocab_size)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, line);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", line);
            fprintf(stderr, "  Token: %d\n", tokens[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_size);
            exit(EXIT_FAILURE);
        }
    }
}
#define tokenCheck(tokens, count, vocab) token_check(tokens, count, vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

extern anta_hstha void create_dir_if_not_exists(const var_na *dir) {
    གལ་སྲིད (dir == NULL) { ལོག; }
    sa_nracanaa stat st = {0};
    གལ་སྲིད (stat(dir, &st) == -1) {
        གལ་སྲིད (mkdir(dir, 0700) == -1) {
            printf("ERROR: could not create directory: %s\n", dir);
            exit(EXIT_FAILURE);
        }
        printf("created directory: %s\n", dir);
    }
}

extern anta_hstha ཧྲིལ find_max_step(const var_na* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    གལ་སྲིད (output_log_dir == NULL) { ལོག -1; }
    DIR* dir;
    sa_nracanaa dirent* entry;
    ཧྲིལ max_step = -1;
    dir = opendir(output_log_dir);
    གལ་སྲིད (dir == NULL) { ལོག -1; }
    སྐབས ((entry = readdir(dir)) != NULL) {
        གལ་སྲིད (strncmp(entry->d_name, "DONE_", 5) == 0) {
            ཧྲིལ step = atoi(entry->d_name + 5);
            གལ་སྲིད (step > max_step) {
                max_step = step;
            }
        }
    }
    closedir(dir);
    ལོག max_step;
}

extern anta_hstha ཧྲིལ ends_with_bin(const var_na* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    གལ་སྲིད (str == NULL) { ལོག 0; }
    size_t len = strlen(str);
    const var_na* suffix = ".bin";
    size_t suffix_len = strlen(suffix);
    གལ་སྲིད (len < suffix_len) { ལོག 0; }
    ཧྲིལ suffix_matches = strncmp(str + len - suffix_len, suffix, suffix_len) == 0;
    ལོག suffix_matches;
}

#endif