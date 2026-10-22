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

extern inline FILE *fopen_check(const char *path, const char *mode, const char *file, 整数 line) {
    FILE *fp = fopen(path, mode);
    如果 (fp == NULL) {
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
    返回 fp;
}

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

extern inline void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, 整数 line) {
    size_t result = fread(ptr, size, nmemb, stream);
    如果 (result != nmemb) {
        如果 (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, line);
        } 否则 如果 (ferror(stream)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, line);
        } 否则 {
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

extern inline void fclose_check(FILE *fp, const char *file, 整数 line) {
    如果 (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

extern inline void sclose_check(整数 sockfd, const char *file, 整数 line) {
    如果 (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#ifdef _WIN32
extern inline void closesocket_check(整数 sockfd, const char *file, 整数 line) {
    如果 (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#endif

extern inline void fseek_check(FILE *fp, long off, 整数 whence, const char *file, 整数 line) {
    如果 (fseek(fp, off, whence) != 0) {
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

extern inline void fwrite_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, 整数 line) {
    size_t result = fwrite(ptr, size, nmemb, stream);
    如果 (result != nmemb) {
        如果 (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, line);
        } 否则 如果 (ferror(stream)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, line);
        } 否则 {
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

extern inline void *malloc_check(size_t size, const char *file, 整数 line) {
    void *ptr = malloc(size);
    如果 (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    返回 ptr;
}

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
extern inline void token_check(const 整数* tokens, 整数 token_count, 整数 vocab_size, const char *file, 整数 line) {
    为(整数 i = 0; i < token_count; i++) {
        如果(!(0 <= tokens[i] && tokens[i] < vocab_size)) {
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

extern inline void create_dir_if_not_exists(const char *dir) {
    如果 (dir == NULL) { 返回; }
    struct stat st = {0};
    如果 (stat(dir, &st) == -1) {
        如果 (mkdir(dir, 0700) == -1) {
            printf("ERROR: could not create directory: %s\n", dir);
            exit(EXIT_FAILURE);
        }
        printf("created directory: %s\n", dir);
    }
}

extern inline 整数 find_max_step(const char* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    如果 (output_log_dir == NULL) { 返回 -1; }
    DIR* dir;
    struct dirent* entry;
    整数 max_step = -1;
    dir = opendir(output_log_dir);
    如果 (dir == NULL) { 返回 -1; }
    当 ((entry = readdir(dir)) != NULL) {
        如果 (strncmp(entry->d_name, "DONE_", 5) == 0) {
            整数 step = atoi(entry->d_name + 5);
            如果 (step > max_step) {
                max_step = step;
            }
        }
    }
    closedir(dir);
    返回 max_step;
}

extern inline 整数 ends_with_bin(const char* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    如果 (str == NULL) { 返回 0; }
    size_t len = strlen(str);
    const char* suffix = ".bin";
    size_t suffix_len = strlen(suffix);
    如果 (len < suffix_len) { 返回 0; }
    整数 suffix_matches = strncmp(str + len - suffix_len, suffix, suffix_len) == 0;
    返回 suffix_matches;
}

#endif