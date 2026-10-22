/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#ifndef UTILS_H
#define UTILS_H

#অন্তর্ভাব <unistd.h>
#অন্তর্ভাব <string.h>
#অন্তর্ভাব <stdio.h>
#অন্তর্ভাব <stdlib.h>
#অন্তর্ভাব <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#ifndef _WIN32
#অন্তর্ভাব <dirent.h>
#অন্তর্ভাব <arpa/inet.h>
#endif

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

বাহ্যিক অন্তঃরেখ FILE *fopen_check(ধ্রুবক অক্ষর *path, ধ্রুবক অক্ষর *mode, ধ্রুবক অক্ষর *file, পূর্ণসংখ্যা সারি) {
    FILE *fp = fopen(path, mode);
    যদি (fp == ফাঁকা) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, সারি);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", সারি);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    ফেরত fp;
}

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

বাহ্যিক অন্তঃরেখ শূন্য fread_check(শূন্য *ptr, size_t আকার_2, size_t nmemb, FILE *stream, ধ্রুবক অক্ষর *file, পূর্ণসংখ্যা সারি) {
    size_t ফলাফল = fread(ptr, আকার_2, nmemb, stream);
    যদি (ফলাফল != nmemb) {
        যদি (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, সারি);
        } অন্যথা যদি (ferror(stream)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, সারি);
        } অন্যথা {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, সারি, nmemb, ফলাফল);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", সারি);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", ফলাফল);
        exit(EXIT_FAILURE);
    }
}

#define freadCheck(ptr, আকার_2, nmemb, stream) fread_check(ptr, আকার_2, nmemb, stream, __FILE__, __LINE__)

বাহ্যিক অন্তঃরেখ শূন্য fclose_check(FILE *fp, ধ্রুবক অক্ষর *file, পূর্ণসংখ্যা সারি) {
    যদি (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, সারি);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", সারি);
        exit(EXIT_FAILURE);
    }
}

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

বাহ্যিক অন্তঃরেখ শূন্য sclose_check(পূর্ণসংখ্যা sockfd, ধ্রুবক অক্ষর *file, পূর্ণসংখ্যা সারি) {
    যদি (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, সারি);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", সারি);
        exit(EXIT_FAILURE);
    }
}

#define scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#ifdef _WIN32
বাহ্যিক অন্তঃরেখ শূন্য closesocket_check(পূর্ণসংখ্যা sockfd, ধ্রুবক অক্ষর *file, পূর্ণসংখ্যা সারি) {
    যদি (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, সারি);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", সারি);
        exit(EXIT_FAILURE);
    }
}

#define closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#endif

বাহ্যিক অন্তঃরেখ শূন্য fseek_check(FILE *fp, long off, পূর্ণসংখ্যা whence, ধ্রুবক অক্ষর *file, পূর্ণসংখ্যা সারি) {
    যদি (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, সারি);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", সারি);
        exit(EXIT_FAILURE);
    }
}

#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

বাহ্যিক অন্তঃরেখ শূন্য fwrite_check(শূন্য *ptr, size_t আকার_2, size_t nmemb, FILE *stream, ধ্রুবক অক্ষর *file, পূর্ণসংখ্যা সারি) {
    size_t ফলাফল = fwrite(ptr, আকার_2, nmemb, stream);
    যদি (ফলাফল != nmemb) {
        যদি (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, সারি);
        } অন্যথা যদি (ferror(stream)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, সারি);
        } অন্যথা {
            fprintf(stderr, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, সারি, nmemb, ফলাফল);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", সারি);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Written elements: %zu\n", ফলাফল);
        exit(EXIT_FAILURE);
    }
}

#define fwriteCheck(ptr, আকার_2, nmemb, stream) fwrite_check(ptr, আকার_2, nmemb, stream, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

বাহ্যিক অন্তঃরেখ শূন্য *malloc_check(size_t আকার_2, ধ্রুবক অক্ষর *file, পূর্ণসংখ্যা সারি) {
    শূন্য *ptr = malloc(আকার_2);
    যদি (ptr == ফাঁকা) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, সারি);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", সারি);
        fprintf(stderr, "  Size: %zu bytes\n", আকার_2);
        exit(EXIT_FAILURE);
    }
    ফেরত ptr;
}

#define mallocCheck(আকার_2) malloc_check(আকার_2, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
বাহ্যিক অন্তঃরেখ শূন্য token_check(ধ্রুবক পূর্ণসংখ্যা* tokens, পূর্ণসংখ্যা token_গুনতি, পূর্ণসংখ্যা vocab_আকার, ধ্রুবক অক্ষর *file, পূর্ণসংখ্যা সারি) {
    জন্য(পূর্ণসংখ্যা i = 0; i < token_গুনতি; i++) {
        যদি(!(0 <= tokens[i] && tokens[i] < vocab_আকার)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, সারি);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", সারি);
            fprintf(stderr, "  Token: %d\n", tokens[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_আকার);
            exit(EXIT_FAILURE);
        }
    }
}
#define tokenCheck(tokens, গুনতি, vocab) token_check(tokens, গুনতি, vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

বাহ্যিক অন্তঃরেখ শূন্য create_dir_if_not_exists(ধ্রুবক অক্ষর *dir) {
    যদি (dir == ফাঁকা) { ফেরত; }
    কাঠামো stat st = {0};
    যদি (stat(dir, &st) == -1) {
        যদি (mkdir(dir, 0700) == -1) {
            printf("ERROR: could not create directory: %s\n", dir);
            exit(EXIT_FAILURE);
        }
        printf("created directory: %s\n", dir);
    }
}

বাহ্যিক অন্তঃরেখ পূর্ণসংখ্যা find_max_step(ধ্রুবক অক্ষর* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    যদি (output_log_dir == ফাঁকা) { ফেরত -1; }
    DIR* dir;
    কাঠামো dirent* entry;
    পূর্ণসংখ্যা max_step = -1;
    dir = opendir(output_log_dir);
    যদি (dir == ফাঁকা) { ফেরত -1; }
    যখন ((entry = readdir(dir)) != ফাঁকা) {
        যদি (strncmp(entry->d_name, "DONE_", 5) == 0) {
            পূর্ণসংখ্যা step = atoi(entry->d_name + 5);
            যদি (step > max_step) {
                max_step = step;
            }
        }
    }
    closedir(dir);
    ফেরত max_step;
}

বাহ্যিক অন্তঃরেখ পূর্ণসংখ্যা ends_with_bin(ধ্রুবক অক্ষর* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    যদি (str == ফাঁকা) { ফেরত 0; }
    size_t len = strlen(str);
    ধ্রুবক অক্ষর* suffix = ".bin";
    size_t suffix_len = strlen(suffix);
    যদি (len < suffix_len) { ফেরত 0; }
    পূর্ণসংখ্যা suffix_matches = strncmp(str + len - suffix_len, suffix, suffix_len) == 0;
    ফেরত suffix_matches;
}

#endif