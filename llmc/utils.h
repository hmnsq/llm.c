/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#ifndef UTILS_H
#define UTILS_H

#અન્તર્ભાવ <unistd.h>
#અન્તર્ભાવ <string.h>
#અન્તર્ભાવ <stdio.h>
#અન્તર્ભાવ <stdlib.h>
#અન્તર્ભાવ <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#ifndef _WIN32
#અન્તર્ભાવ <dirent.h>
#અન્તર્ભાવ <arpa/inet.h>
#endif

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

બાહ્ય અંતઃરેખ FILE *fopen_check(અચલ અક્ષર *path, અચલ અક્ષર *mode, અચલ અક્ષર *file, પૂર્ણાંક લીટી) {
    FILE *fp = fopen(path, mode);
    જો (fp == ખાલી) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, લીટી);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", લીટી);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    પાછા fp;
}

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

બાહ્ય અંતઃરેખ શૂન્ય fread_check(શૂન્ય *ptr, size_t size, size_t nmemb, FILE *stream, અચલ અક્ષર *file, પૂર્ણાંક લીટી) {
    size_t result = fread(ptr, size, nmemb, stream);
    જો (result != nmemb) {
        જો (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, લીટી);
        } નહીંતર જો (ferror(stream)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, લીટી);
        } નહીંતર {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, લીટી, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", લીટી);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

બાહ્ય અંતઃરેખ શૂન્ય fclose_check(FILE *fp, અચલ અક્ષર *file, પૂર્ણાંક લીટી) {
    જો (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, લીટી);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", લીટી);
        exit(EXIT_FAILURE);
    }
}

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

બાહ્ય અંતઃરેખ શૂન્ય sclose_check(પૂર્ણાંક sockfd, અચલ અક્ષર *file, પૂર્ણાંક લીટી) {
    જો (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, લીટી);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", લીટી);
        exit(EXIT_FAILURE);
    }
}

#define scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#ifdef _WIN32
બાહ્ય અંતઃરેખ શૂન્ય closesocket_check(પૂર્ણાંક sockfd, અચલ અક્ષર *file, પૂર્ણાંક લીટી) {
    જો (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, લીટી);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", લીટી);
        exit(EXIT_FAILURE);
    }
}

#define closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#endif

બાહ્ય અંતઃરેખ શૂન્ય fseek_check(FILE *fp, long off, પૂર્ણાંક whence, અચલ અક્ષર *file, પૂર્ણાંક લીટી) {
    જો (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, લીટી);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", લીટી);
        exit(EXIT_FAILURE);
    }
}

#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

બાહ્ય અંતઃરેખ શૂન્ય fwrite_check(શૂન્ય *ptr, size_t size, size_t nmemb, FILE *stream, અચલ અક્ષર *file, પૂર્ણાંક લીટી) {
    size_t result = fwrite(ptr, size, nmemb, stream);
    જો (result != nmemb) {
        જો (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, લીટી);
        } નહીંતર જો (ferror(stream)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, લીટી);
        } નહીંતર {
            fprintf(stderr, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, લીટી, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", લીટી);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Written elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define fwriteCheck(ptr, size, nmemb, stream) fwrite_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

બાહ્ય અંતઃરેખ શૂન્ય *malloc_check(size_t size, અચલ અક્ષર *file, પૂર્ણાંક લીટી) {
    શૂન્ય *ptr = malloc(size);
    જો (ptr == ખાલી) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, લીટી);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", લીટી);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    પાછા ptr;
}

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
બાહ્ય અંતઃરેખ શૂન્ય token_check(અચલ પૂર્ણાંક* tokens, પૂર્ણાંક token_ગણતરી, પૂર્ણાંક vocab_size, અચલ અક્ષર *file, પૂર્ણાંક લીટી) {
    માટે(પૂર્ણાંક i = 0; i < token_ગણતરી; i++) {
        જો(!(0 <= tokens[i] && tokens[i] < vocab_size)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, લીટી);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", લીટી);
            fprintf(stderr, "  Token: %d\n", tokens[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_size);
            exit(EXIT_FAILURE);
        }
    }
}
#define tokenCheck(tokens, ગણતરી, vocab) token_check(tokens, ગણતરી, vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

બાહ્ય અંતઃરેખ શૂન્ય create_dir_if_not_exists(અચલ અક્ષર *dir) {
    જો (dir == ખાલી) { પાછા; }
    સંરચના stat st = {0};
    જો (stat(dir, &st) == -1) {
        જો (mkdir(dir, 0700) == -1) {
            printf("ERROR: could not create directory: %s\n", dir);
            exit(EXIT_FAILURE);
        }
        printf("created directory: %s\n", dir);
    }
}

બાહ્ય અંતઃરેખ પૂર્ણાંક find_max_step(અચલ અક્ષર* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    જો (output_log_dir == ખાલી) { પાછા -1; }
    DIR* dir;
    સંરચના dirent* entry;
    પૂર્ણાંક max_step = -1;
    dir = opendir(output_log_dir);
    જો (dir == ખાલી) { પાછા -1; }
    જ્યારે ((entry = readdir(dir)) != ખાલી) {
        જો (strncmp(entry->d_name, "DONE_", 5) == 0) {
            પૂર્ણાંક step = atoi(entry->d_name + 5);
            જો (step > max_step) {
                max_step = step;
            }
        }
    }
    closedir(dir);
    પાછા max_step;
}

બાહ્ય અંતઃરેખ પૂર્ણાંક ends_with_bin(અચલ અક્ષર* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    જો (str == ખાલી) { પાછા 0; }
    size_t len = strlen(str);
    અચલ અક્ષર* suffix = ".bin";
    size_t suffix_len = strlen(suffix);
    જો (len < suffix_len) { પાછા 0; }
    પૂર્ણાંક suffix_matches = strncmp(str + len - suffix_len, suffix, suffix_len) == 0;
    પાછા suffix_matches;
}

#endif