/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#ifndef UTILS_H
#define UTILS_H

#ಅನ್ತರ್ಭಾವ <unistd.h>
#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#ifndef _WIN32
#ಅನ್ತರ್ಭಾವ <dirent.h>
#ಅನ್ತರ್ಭಾವ <arpa/inet.h>
#endif

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

ಬಾಹ್ಯ ಅನ್ತಃಸ್ಥ FILE *fopen_check(ಧ್ರುವ ವರ್ಣ *path, ಧ್ರುವ ವರ್ಣ *mode, ಧ್ರುವ ವರ್ಣ *file, ಪೂರ್ಣಾಙ್ಕ ಸಾಲು) {
    FILE *fp = fopen(path, mode);
    ಯದಿ (fp == ರಿಕ್ತ) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, ಸಾಲು);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ಸಾಲು);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    ಪ್ರತ್ಯಾಗಮ fp;
}

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

ಬಾಹ್ಯ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ fread_check(ಶೂನ್ಯ *ptr, size_t size, size_t nmemb, FILE *stream, ಧ್ರುವ ವರ್ಣ *file, ಪೂರ್ಣಾಙ್ಕ ಸಾಲು) {
    size_t result = fread(ptr, size, nmemb, stream);
    ಯದಿ (result != nmemb) {
        ಯದಿ (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, ಸಾಲು);
        } ಅನ್ಯಥಾ ಯದಿ (ferror(stream)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, ಸಾಲು);
        } ಅನ್ಯಥಾ {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, ಸಾಲು, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ಸಾಲು);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

ಬಾಹ್ಯ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ fclose_check(FILE *fp, ಧ್ರುವ ವರ್ಣ *file, ಪೂರ್ಣಾಙ್ಕ ಸಾಲು) {
    ಯದಿ (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, ಸಾಲು);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ಸಾಲು);
        exit(EXIT_FAILURE);
    }
}

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

ಬಾಹ್ಯ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ sclose_check(ಪೂರ್ಣಾಙ್ಕ sockfd, ಧ್ರುವ ವರ್ಣ *file, ಪೂರ್ಣಾಙ್ಕ ಸಾಲು) {
    ಯದಿ (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, ಸಾಲು);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ಸಾಲು);
        exit(EXIT_FAILURE);
    }
}

#define scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#ifdef _WIN32
ಬಾಹ್ಯ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ closesocket_check(ಪೂರ್ಣಾಙ್ಕ sockfd, ಧ್ರುವ ವರ್ಣ *file, ಪೂರ್ಣಾಙ್ಕ ಸಾಲು) {
    ಯದಿ (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, ಸಾಲು);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ಸಾಲು);
        exit(EXIT_FAILURE);
    }
}

#define closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#endif

ಬಾಹ್ಯ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ fseek_check(FILE *fp, long off, ಪೂರ್ಣಾಙ್ಕ whence, ಧ್ರುವ ವರ್ಣ *file, ಪೂರ್ಣಾಙ್ಕ ಸಾಲು) {
    ಯದಿ (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, ಸಾಲು);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", ಸಾಲು);
        exit(EXIT_FAILURE);
    }
}

#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

ಬಾಹ್ಯ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ fwrite_check(ಶೂನ್ಯ *ptr, size_t size, size_t nmemb, FILE *stream, ಧ್ರುವ ವರ್ಣ *file, ಪೂರ್ಣಾಙ್ಕ ಸಾಲು) {
    size_t result = fwrite(ptr, size, nmemb, stream);
    ಯದಿ (result != nmemb) {
        ಯದಿ (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, ಸಾಲು);
        } ಅನ್ಯಥಾ ಯದಿ (ferror(stream)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, ಸಾಲು);
        } ಅನ್ಯಥಾ {
            fprintf(stderr, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, ಸಾಲು, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ಸಾಲು);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Written elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define fwriteCheck(ptr, size, nmemb, stream) fwrite_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

ಬಾಹ್ಯ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ *malloc_check(size_t size, ಧ್ರುವ ವರ್ಣ *file, ಪೂರ್ಣಾಙ್ಕ ಸಾಲು) {
    ಶೂನ್ಯ *ptr = malloc(size);
    ಯದಿ (ptr == ರಿಕ್ತ) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, ಸಾಲು);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ಸಾಲು);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    ಪ್ರತ್ಯಾಗಮ ptr;
}

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
ಬಾಹ್ಯ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ token_check(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ* tokens, ಪೂರ್ಣಾಙ್ಕ token_ಎಣಿಕೆ, ಪೂರ್ಣಾಙ್ಕ vocab_size, ಧ್ರುವ ವರ್ಣ *file, ಪೂರ್ಣಾಙ್ಕ ಸಾಲು) {
    ಪ್ರತಿ(ಪೂರ್ಣಾಙ್ಕ i = 0; i < token_ಎಣಿಕೆ; i++) {
        ಯದಿ(!(0 <= tokens[i] && tokens[i] < vocab_size)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, ಸಾಲು);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", ಸಾಲು);
            fprintf(stderr, "  Token: %d\n", tokens[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_size);
            exit(EXIT_FAILURE);
        }
    }
}
#define tokenCheck(tokens, ಎಣಿಕೆ, vocab) token_check(tokens, ಎಣಿಕೆ, vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

ಬಾಹ್ಯ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ create_dir_if_not_exists(ಧ್ರುವ ವರ್ಣ *dir) {
    ಯದಿ (dir == ರಿಕ್ತ) { ಪ್ರತ್ಯಾಗಮ; }
    ಸಂರಚನಾ stat st = {0};
    ಯದಿ (stat(dir, &st) == -1) {
        ಯದಿ (mkdir(dir, 0700) == -1) {
            printf("ERROR: could not create directory: %s\n", dir);
            exit(EXIT_FAILURE);
        }
        printf("created directory: %s\n", dir);
    }
}

ಬಾಹ್ಯ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ find_max_step(ಧ್ರುವ ವರ್ಣ* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    ಯದಿ (output_log_dir == ರಿಕ್ತ) { ಪ್ರತ್ಯಾಗಮ -1; }
    DIR* dir;
    ಸಂರಚನಾ dirent* entry;
    ಪೂರ್ಣಾಙ್ಕ max_step = -1;
    dir = opendir(output_log_dir);
    ಯದಿ (dir == ರಿಕ್ತ) { ಪ್ರತ್ಯಾಗಮ -1; }
    ಯಾವತ್ ((entry = readdir(dir)) != ರಿಕ್ತ) {
        ಯದಿ (strncmp(entry->d_name, "DONE_", 5) == 0) {
            ಪೂರ್ಣಾಙ್ಕ step = atoi(entry->d_name + 5);
            ಯದಿ (step > max_step) {
                max_step = step;
            }
        }
    }
    closedir(dir);
    ಪ್ರತ್ಯಾಗಮ max_step;
}

ಬಾಹ್ಯ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ ends_with_bin(ಧ್ರುವ ವರ್ಣ* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    ಯದಿ (str == ರಿಕ್ತ) { ಪ್ರತ್ಯಾಗಮ 0; }
    size_t len = strlen(str);
    ಧ್ರುವ ವರ್ಣ* suffix = ".bin";
    size_t suffix_len = strlen(suffix);
    ಯದಿ (len < suffix_len) { ಪ್ರತ್ಯಾಗಮ 0; }
    ಪೂರ್ಣಾಙ್ಕ suffix_matches = strncmp(str + len - suffix_len, suffix, suffix_len) == 0;
    ಪ್ರತ್ಯಾಗಮ suffix_matches;
}

#endif