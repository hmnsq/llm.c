/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#ifndef UTILS_H
#define UTILS_H

#ضمن <unistd.h>
#ضمن <string.h>
#ضمن <stdio.h>
#ضمن <stdlib.h>
#ضمن <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#ifndef _WIN32
#ضمن <dirent.h>
#ضمن <arpa/inet.h>
#endif

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

خارجي مضمن FILE *fopen_check(ثابت حرف *path, ثابت حرف *mode, ثابت حرف *file, صحيح هێڵ) {
    FILE *fp = fopen(path, mode);
    إذا (fp == عدم) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, هێڵ);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", هێڵ);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    أرجع fp;
}

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

خارجي مضمن فراغ fread_check(فراغ *ptr, size_t size, size_t nmemb, FILE *stream, ثابت حرف *file, صحيح هێڵ) {
    size_t result = fread(ptr, size, nmemb, stream);
    إذا (result != nmemb) {
        إذا (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, هێڵ);
        } وإلا إذا (ferror(stream)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, هێڵ);
        } وإلا {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, هێڵ, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", هێڵ);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

خارجي مضمن فراغ fclose_check(FILE *fp, ثابت حرف *file, صحيح هێڵ) {
    إذا (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, هێڵ);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", هێڵ);
        exit(EXIT_FAILURE);
    }
}

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

خارجي مضمن فراغ sclose_check(صحيح sockfd, ثابت حرف *file, صحيح هێڵ) {
    إذا (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, هێڵ);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", هێڵ);
        exit(EXIT_FAILURE);
    }
}

#define scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#ifdef _WIN32
خارجي مضمن فراغ closesocket_check(صحيح sockfd, ثابت حرف *file, صحيح هێڵ) {
    إذا (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, هێڵ);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", هێڵ);
        exit(EXIT_FAILURE);
    }
}

#define closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#endif

خارجي مضمن فراغ fseek_check(FILE *fp, long off, صحيح whence, ثابت حرف *file, صحيح هێڵ) {
    إذا (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, هێڵ);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", هێڵ);
        exit(EXIT_FAILURE);
    }
}

#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

خارجي مضمن فراغ fwrite_check(فراغ *ptr, size_t size, size_t nmemb, FILE *stream, ثابت حرف *file, صحيح هێڵ) {
    size_t result = fwrite(ptr, size, nmemb, stream);
    إذا (result != nmemb) {
        إذا (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, هێڵ);
        } وإلا إذا (ferror(stream)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, هێڵ);
        } وإلا {
            fprintf(stderr, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, هێڵ, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", هێڵ);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Written elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define fwriteCheck(ptr, size, nmemb, stream) fwrite_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

خارجي مضمن فراغ *malloc_check(size_t size, ثابت حرف *file, صحيح هێڵ) {
    فراغ *ptr = malloc(size);
    إذا (ptr == عدم) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, هێڵ);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", هێڵ);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    أرجع ptr;
}

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
خارجي مضمن فراغ token_check(ثابت صحيح* tokens, صحيح token_ژماردن, صحيح vocab_size, ثابت حرف *file, صحيح هێڵ) {
    لكل(صحيح i = 0; i < token_ژماردن; i++) {
        إذا(!(0 <= tokens[i] && tokens[i] < vocab_size)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, هێڵ);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", هێڵ);
            fprintf(stderr, "  Token: %d\n", tokens[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_size);
            exit(EXIT_FAILURE);
        }
    }
}
#define tokenCheck(tokens, ژماردن, vocab) token_check(tokens, ژماردن, vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

خارجي مضمن فراغ create_dir_if_not_exists(ثابت حرف *dir) {
    إذا (dir == عدم) { أرجع; }
    بنية stat st = {0};
    إذا (stat(dir, &st) == -1) {
        إذا (mkdir(dir, 0700) == -1) {
            printf("ERROR: could not create directory: %s\n", dir);
            exit(EXIT_FAILURE);
        }
        printf("created directory: %s\n", dir);
    }
}

خارجي مضمن صحيح find_max_step(ثابت حرف* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    إذا (output_log_dir == عدم) { أرجع -1; }
    DIR* dir;
    بنية dirent* entry;
    صحيح max_step = -1;
    dir = opendir(output_log_dir);
    إذا (dir == عدم) { أرجع -1; }
    طالما ((entry = readdir(dir)) != عدم) {
        إذا (strncmp(entry->d_name, "DONE_", 5) == 0) {
            صحيح step = atoi(entry->d_name + 5);
            إذا (step > max_step) {
                max_step = step;
            }
        }
    }
    closedir(dir);
    أرجع max_step;
}

خارجي مضمن صحيح ends_with_bin(ثابت حرف* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    إذا (str == عدم) { أرجع 0; }
    size_t len = strlen(str);
    ثابت حرف* suffix = ".bin";
    size_t suffix_len = strlen(suffix);
    إذا (len < suffix_len) { أرجع 0; }
    صحيح suffix_matches = strncmp(str + len - suffix_len, suffix, suffix_len) == 0;
    أرجع suffix_matches;
}

#endif