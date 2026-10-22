/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#ifndef UTILS_H
#define UTILS_H

#അന്തര്ഭാവ <unistd.h>
#അന്തര്ഭാവ <string.h>
#അന്തര്ഭാവ <stdio.h>
#അന്തര്ഭാവ <stdlib.h>
#അന്തര്ഭാവ <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#ifndef _WIN32
#അന്തര്ഭാവ <dirent.h>
#അന്തര്ഭാവ <arpa/inet.h>
#endif

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

ബാഹ്യ വരിയിൽ FILE *fopen_check(സ്ഥിരം അക്ഷരം *path, സ്ഥിരം അക്ഷരം *mode, സ്ഥിരം അക്ഷരം *file, പൂർണ്ണസംഖ്യ വരി) {
    FILE *fp = fopen(path, mode);
    എങ്കിൽ (fp == ശൂന്യ) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, വരി);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", വരി);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    തിരികെ fp;
}

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

ബാഹ്യ വരിയിൽ ശൂന്യം fread_check(ശൂന്യം *ptr, size_t size, size_t nmemb, FILE *stream, സ്ഥിരം അക്ഷരം *file, പൂർണ്ണസംഖ്യ വരി) {
    size_t result = fread(ptr, size, nmemb, stream);
    എങ്കിൽ (result != nmemb) {
        എങ്കിൽ (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, വരി);
        } അല്ലെങ്കിൽ എങ്കിൽ (ferror(stream)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, വരി);
        } അല്ലെങ്കിൽ {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, വരി, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", വരി);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

ബാഹ്യ വരിയിൽ ശൂന്യം fclose_check(FILE *fp, സ്ഥിരം അക്ഷരം *file, പൂർണ്ണസംഖ്യ വരി) {
    എങ്കിൽ (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, വരി);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", വരി);
        exit(EXIT_FAILURE);
    }
}

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

ബാഹ്യ വരിയിൽ ശൂന്യം sclose_check(പൂർണ്ണസംഖ്യ sockfd, സ്ഥിരം അക്ഷരം *file, പൂർണ്ണസംഖ്യ വരി) {
    എങ്കിൽ (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, വരി);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", വരി);
        exit(EXIT_FAILURE);
    }
}

#define scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#ifdef _WIN32
ബാഹ്യ വരിയിൽ ശൂന്യം closesocket_check(പൂർണ്ണസംഖ്യ sockfd, സ്ഥിരം അക്ഷരം *file, പൂർണ്ണസംഖ്യ വരി) {
    എങ്കിൽ (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, വരി);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", വരി);
        exit(EXIT_FAILURE);
    }
}

#define closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#endif

ബാഹ്യ വരിയിൽ ശൂന്യം fseek_check(FILE *fp, long off, പൂർണ്ണസംഖ്യ whence, സ്ഥിരം അക്ഷരം *file, പൂർണ്ണസംഖ്യ വരി) {
    എങ്കിൽ (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, വരി);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", വരി);
        exit(EXIT_FAILURE);
    }
}

#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

ബാഹ്യ വരിയിൽ ശൂന്യം fwrite_check(ശൂന്യം *ptr, size_t size, size_t nmemb, FILE *stream, സ്ഥിരം അക്ഷരം *file, പൂർണ്ണസംഖ്യ വരി) {
    size_t result = fwrite(ptr, size, nmemb, stream);
    എങ്കിൽ (result != nmemb) {
        എങ്കിൽ (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, വരി);
        } അല്ലെങ്കിൽ എങ്കിൽ (ferror(stream)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, വരി);
        } അല്ലെങ്കിൽ {
            fprintf(stderr, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, വരി, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", വരി);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Written elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define fwriteCheck(ptr, size, nmemb, stream) fwrite_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

ബാഹ്യ വരിയിൽ ശൂന്യം *malloc_check(size_t size, സ്ഥിരം അക്ഷരം *file, പൂർണ്ണസംഖ്യ വരി) {
    ശൂന്യം *ptr = malloc(size);
    എങ്കിൽ (ptr == ശൂന്യ) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, വരി);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", വരി);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    തിരികെ ptr;
}

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
ബാഹ്യ വരിയിൽ ശൂന്യം token_check(സ്ഥിരം പൂർണ്ണസംഖ്യ* tokens, പൂർണ്ണസംഖ്യ token_എണ്ണം, പൂർണ്ണസംഖ്യ vocab_size, സ്ഥിരം അക്ഷരം *file, പൂർണ്ണസംഖ്യ വരി) {
    വേണ്ടി(പൂർണ്ണസംഖ്യ i = 0; i < token_എണ്ണം; i++) {
        എങ്കിൽ(!(0 <= tokens[i] && tokens[i] < vocab_size)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, വരി);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", വരി);
            fprintf(stderr, "  Token: %d\n", tokens[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_size);
            exit(EXIT_FAILURE);
        }
    }
}
#define tokenCheck(tokens, എണ്ണം, vocab) token_check(tokens, എണ്ണം, vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

ബാഹ്യ വരിയിൽ ശൂന്യം create_dir_if_not_exists(സ്ഥിരം അക്ഷരം *dir) {
    എങ്കിൽ (dir == ശൂന്യ) { തിരികെ; }
    ഘടന stat st = {0};
    എങ്കിൽ (stat(dir, &st) == -1) {
        എങ്കിൽ (mkdir(dir, 0700) == -1) {
            printf("ERROR: could not create directory: %s\n", dir);
            exit(EXIT_FAILURE);
        }
        printf("created directory: %s\n", dir);
    }
}

ബാഹ്യ വരിയിൽ പൂർണ്ണസംഖ്യ find_max_step(സ്ഥിരം അക്ഷരം* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    എങ്കിൽ (output_log_dir == ശൂന്യ) { തിരികെ -1; }
    DIR* dir;
    ഘടന dirent* entry;
    പൂർണ്ണസംഖ്യ max_step = -1;
    dir = opendir(output_log_dir);
    എങ്കിൽ (dir == ശൂന്യ) { തിരികെ -1; }
    ആകുമ്പോൾ ((entry = readdir(dir)) != ശൂന്യ) {
        എങ്കിൽ (strncmp(entry->d_name, "DONE_", 5) == 0) {
            പൂർണ്ണസംഖ്യ step = atoi(entry->d_name + 5);
            എങ്കിൽ (step > max_step) {
                max_step = step;
            }
        }
    }
    closedir(dir);
    തിരികെ max_step;
}

ബാഹ്യ വരിയിൽ പൂർണ്ണസംഖ്യ ends_with_bin(സ്ഥിരം അക്ഷരം* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    എങ്കിൽ (str == ശൂന്യ) { തിരികെ 0; }
    size_t len = strlen(str);
    സ്ഥിരം അക്ഷരം* suffix = ".bin";
    size_t suffix_len = strlen(suffix);
    എങ്കിൽ (len < suffix_len) { തിരികെ 0; }
    പൂർണ്ണസംഖ്യ suffix_matches = strncmp(str + len - suffix_len, suffix, suffix_len) == 0;
    തിരികെ suffix_matches;
}

#endif