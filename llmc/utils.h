/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#ifndef UTILS_H
#define UTILS_H

#অন্তর্ভাৱ <unistd.h>
#অন্তর্ভাৱ <string.h>
#অন্তর্ভাৱ <stdio.h>
#অন্তর্ভাৱ <stdlib.h>
#অন্তর্ভাৱ <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#ifndef _WIN32
#অন্তর্ভাৱ <dirent.h>
#অন্তর্ভাৱ <arpa/inet.h>
#endif

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

বাহ্য অন্তঃস্থ FILE *fopen_check(ধ্রুৱ ৱর্ণ *path, ধ্রুৱ ৱর্ণ *mode, ধ্রুৱ ৱর্ণ *file, পূর্ণাঙ্ক সাৰি) {
    FILE *fp = fopen(path, mode);
    যদি (fp == রিক্ত) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, সাৰি);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", সাৰি);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    ঘূৰাই fp;
}

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

বাহ্য অন্তঃস্থ শূন্য fread_check(শূন্য *ptr, size_t size, size_t nmemb, FILE *stream, ধ্রুৱ ৱর্ণ *file, পূর্ণাঙ্ক সাৰি) {
    size_t result = fread(ptr, size, nmemb, stream);
    যদি (result != nmemb) {
        যদি (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, সাৰি);
        } অন্যথা যদি (ferror(stream)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, সাৰি);
        } অন্যথা {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, সাৰি, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", সাৰি);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

বাহ্য অন্তঃস্থ শূন্য fclose_check(FILE *fp, ধ্রুৱ ৱর্ণ *file, পূর্ণাঙ্ক সাৰি) {
    যদি (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, সাৰি);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", সাৰি);
        exit(EXIT_FAILURE);
    }
}

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

বাহ্য অন্তঃস্থ শূন্য sclose_check(পূর্ণাঙ্ক sockfd, ধ্রুৱ ৱর্ণ *file, পূর্ণাঙ্ক সাৰি) {
    যদি (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, সাৰি);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", সাৰি);
        exit(EXIT_FAILURE);
    }
}

#define scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#ifdef _WIN32
বাহ্য অন্তঃস্থ শূন্য closesocket_check(পূর্ণাঙ্ক sockfd, ধ্রুৱ ৱর্ণ *file, পূর্ণাঙ্ক সাৰি) {
    যদি (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, সাৰি);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", সাৰি);
        exit(EXIT_FAILURE);
    }
}

#define closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#endif

বাহ্য অন্তঃস্থ শূন্য fseek_check(FILE *fp, long off, পূর্ণাঙ্ক whence, ধ্রুৱ ৱর্ণ *file, পূর্ণাঙ্ক সাৰি) {
    যদি (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, সাৰি);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", সাৰি);
        exit(EXIT_FAILURE);
    }
}

#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

বাহ্য অন্তঃস্থ শূন্য fwrite_check(শূন্য *ptr, size_t size, size_t nmemb, FILE *stream, ধ্রুৱ ৱর্ণ *file, পূর্ণাঙ্ক সাৰি) {
    size_t result = fwrite(ptr, size, nmemb, stream);
    যদি (result != nmemb) {
        যদি (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, সাৰি);
        } অন্যথা যদি (ferror(stream)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, সাৰি);
        } অন্যথা {
            fprintf(stderr, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, সাৰি, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", সাৰি);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Written elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define fwriteCheck(ptr, size, nmemb, stream) fwrite_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

বাহ্য অন্তঃস্থ শূন্য *malloc_check(size_t size, ধ্রুৱ ৱর্ণ *file, পূর্ণাঙ্ক সাৰি) {
    শূন্য *ptr = malloc(size);
    যদি (ptr == রিক্ত) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, সাৰি);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", সাৰি);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    ঘূৰাই ptr;
}

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
বাহ্য অন্তঃস্থ শূন্য token_check(ধ্রুৱ পূর্ণাঙ্ক* tokens, পূর্ণাঙ্ক token_গুনতি, পূর্ণাঙ্ক vocab_size, ধ্রুৱ ৱর্ণ *file, পূর্ণাঙ্ক সাৰি) {
    বাবে(পূর্ণাঙ্ক i = 0; i < token_গুনতি; i++) {
        যদি(!(0 <= tokens[i] && tokens[i] < vocab_size)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, সাৰি);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", সাৰি);
            fprintf(stderr, "  Token: %d\n", tokens[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_size);
            exit(EXIT_FAILURE);
        }
    }
}
#define tokenCheck(tokens, গুনতি, vocab) token_check(tokens, গুনতি, vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

বাহ্য অন্তঃস্থ শূন্য create_dir_if_not_exists(ধ্রুৱ ৱর্ণ *dir) {
    যদি (dir == রিক্ত) { ঘূৰাই; }
    সংরচনা stat st = {0};
    যদি (stat(dir, &st) == -1) {
        যদি (mkdir(dir, 0700) == -1) {
            printf("ERROR: could not create directory: %s\n", dir);
            exit(EXIT_FAILURE);
        }
        printf("created directory: %s\n", dir);
    }
}

বাহ্য অন্তঃস্থ পূর্ণাঙ্ক find_max_step(ধ্রুৱ ৱর্ণ* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    যদি (output_log_dir == রিক্ত) { ঘূৰাই -1; }
    DIR* dir;
    সংরচনা dirent* entry;
    পূর্ণাঙ্ক max_step = -1;
    dir = opendir(output_log_dir);
    যদি (dir == রিক্ত) { ঘূৰাই -1; }
    যেতিয়ালৈ ((entry = readdir(dir)) != রিক্ত) {
        যদি (strncmp(entry->d_name, "DONE_", 5) == 0) {
            পূর্ণাঙ্ক step = atoi(entry->d_name + 5);
            যদি (step > max_step) {
                max_step = step;
            }
        }
    }
    closedir(dir);
    ঘূৰাই max_step;
}

বাহ্য অন্তঃস্থ পূর্ণাঙ্ক ends_with_bin(ধ্রুৱ ৱর্ণ* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    যদি (str == রিক্ত) { ঘূৰাই 0; }
    size_t len = strlen(str);
    ধ্রুৱ ৱর্ণ* suffix = ".bin";
    size_t suffix_len = strlen(suffix);
    যদি (len < suffix_len) { ঘূৰাই 0; }
    পূর্ণাঙ্ক suffix_matches = strncmp(str + len - suffix_len, suffix, suffix_len) == 0;
    ঘূৰাই suffix_matches;
}

#endif