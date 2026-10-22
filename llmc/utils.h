/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#ifndef UTILS_H
#define UTILS_H

#अन्तर्भाव <unistd.h>
#अन्तर्भाव <string.h>
#अन्तर्भाव <stdio.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#ifndef _WIN32
#अन्तर्भाव <dirent.h>
#अन्तर्भाव <arpa/inet.h>
#endif

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

बाह्य अन्तःस्थ FILE *fopen_check(ध्रुव वर्ण *path, ध्रुव वर्ण *mode, ध्रुव वर्ण *file, पूर्णाङ्क पङ्क्तिः) {
    FILE *fp = fopen(path, mode);
    सचे (fp == रिक्त) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, पङ्क्तिः);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", पङ्क्तिः);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    पटिनिवत्त fp;
}

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

बाह्य अन्तःस्थ शून्य fread_check(शून्य *ptr, size_t size, size_t nmemb, FILE *stream, ध्रुव वर्ण *file, पूर्णाङ्क पङ्क्तिः) {
    size_t result = fread(ptr, size, nmemb, stream);
    सचे (result != nmemb) {
        सचे (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, पङ्क्तिः);
        } अञ्ञथा सचे (ferror(stream)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, पङ्क्तिः);
        } अञ्ञथा {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, पङ्क्तिः, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", पङ्क्तिः);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

बाह्य अन्तःस्थ शून्य fclose_check(FILE *fp, ध्रुव वर्ण *file, पूर्णाङ्क पङ्क्तिः) {
    सचे (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, पङ्क्तिः);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", पङ्क्तिः);
        exit(EXIT_FAILURE);
    }
}

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

बाह्य अन्तःस्थ शून्य sclose_check(पूर्णाङ्क sockfd, ध्रुव वर्ण *file, पूर्णाङ्क पङ्क्तिः) {
    सचे (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, पङ्क्तिः);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", पङ्क्तिः);
        exit(EXIT_FAILURE);
    }
}

#define scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#ifdef _WIN32
बाह्य अन्तःस्थ शून्य closesocket_check(पूर्णाङ्क sockfd, ध्रुव वर्ण *file, पूर्णाङ्क पङ्क्तिः) {
    सचे (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, पङ्क्तिः);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", पङ्क्तिः);
        exit(EXIT_FAILURE);
    }
}

#define closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#endif

बाह्य अन्तःस्थ शून्य fseek_check(FILE *fp, long off, पूर्णाङ्क whence, ध्रुव वर्ण *file, पूर्णाङ्क पङ्क्तिः) {
    सचे (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, पङ्क्तिः);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", पङ्क्तिः);
        exit(EXIT_FAILURE);
    }
}

#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

बाह्य अन्तःस्थ शून्य fwrite_check(शून्य *ptr, size_t size, size_t nmemb, FILE *stream, ध्रुव वर्ण *file, पूर्णाङ्क पङ्क्तिः) {
    size_t result = fwrite(ptr, size, nmemb, stream);
    सचे (result != nmemb) {
        सचे (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, पङ्क्तिः);
        } अञ्ञथा सचे (ferror(stream)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, पङ्क्तिः);
        } अञ्ञथा {
            fprintf(stderr, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, पङ्क्तिः, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", पङ्क्तिः);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Written elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define fwriteCheck(ptr, size, nmemb, stream) fwrite_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

बाह्य अन्तःस्थ शून्य *malloc_check(size_t size, ध्रुव वर्ण *file, पूर्णाङ्क पङ्क्तिः) {
    शून्य *ptr = malloc(size);
    सचे (ptr == रिक्त) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, पङ्क्तिः);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", पङ्क्तिः);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    पटिनिवत्त ptr;
}

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
बाह्य अन्तःस्थ शून्य token_check(ध्रुव पूर्णाङ्क* tokens, पूर्णाङ्क token_गणनम्, पूर्णाङ्क vocab_size, ध्रुव वर्ण *file, पूर्णाङ्क पङ्क्तिः) {
    अत्थाय(पूर्णाङ्क i = 0; i < token_गणनम्; i++) {
        सचे(!(0 <= tokens[i] && tokens[i] < vocab_size)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, पङ्क्तिः);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", पङ्क्तिः);
            fprintf(stderr, "  Token: %d\n", tokens[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_size);
            exit(EXIT_FAILURE);
        }
    }
}
#define tokenCheck(tokens, गणनम्, vocab) token_check(tokens, गणनम्, vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

बाह्य अन्तःस्थ शून्य create_dir_if_not_exists(ध्रुव वर्ण *dir) {
    सचे (dir == रिक्त) { पटिनिवत्त; }
    संरचना stat st = {0};
    सचे (stat(dir, &st) == -1) {
        सचे (mkdir(dir, 0700) == -1) {
            printf("ERROR: could not create directory: %s\n", dir);
            exit(EXIT_FAILURE);
        }
        printf("created directory: %s\n", dir);
    }
}

बाह्य अन्तःस्थ पूर्णाङ्क find_max_step(ध्रुव वर्ण* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    सचे (output_log_dir == रिक्त) { पटिनिवत्त -1; }
    DIR* dir;
    संरचना dirent* entry;
    पूर्णाङ्क max_step = -1;
    dir = opendir(output_log_dir);
    सचे (dir == रिक्त) { पटिनिवत्त -1; }
    याव ((entry = readdir(dir)) != रिक्त) {
        सचे (strncmp(entry->d_name, "DONE_", 5) == 0) {
            पूर्णाङ्क step = atoi(entry->d_name + 5);
            सचे (step > max_step) {
                max_step = step;
            }
        }
    }
    closedir(dir);
    पटिनिवत्त max_step;
}

बाह्य अन्तःस्थ पूर्णाङ्क ends_with_bin(ध्रुव वर्ण* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    सचे (str == रिक्त) { पटिनिवत्त 0; }
    size_t len = strlen(str);
    ध्रुव वर्ण* suffix = ".bin";
    size_t suffix_len = strlen(suffix);
    सचे (len < suffix_len) { पटिनिवत्त 0; }
    पूर्णाङ्क suffix_matches = strncmp(str + len - suffix_len, suffix, suffix_len) == 0;
    पटिनिवत्त suffix_matches;
}

#endif