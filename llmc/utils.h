/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ UTILS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ UTILS_H

#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ _WIN32
#ਅਨ੍ਤਰ੍ਭਾਵ <dirent.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <arpa/inet.h>
#ਜੇ_ਅੰਤ

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

ਬਾਹਰੀ ਸਤਰਅੰਦਰ FILE *fopen_check(ਅਚਲ ਅੱਖਰ *ਰਸਤਾ, ਅਚਲ ਅੱਖਰ *ਬਹੁਲਕ, ਅਚਲ ਅੱਖਰ *file, ਪੂਰਨ ਸਤਰ) {
    FILE *fp = fopen(ਰਸਤਾ, ਬਹੁਲਕ);
    ਜੇ (fp == ਨਿਰਾ) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", ਰਸਤਾ, file, ਸਤਰ);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ਸਤਰ);
        fprintf(stderr, "  Path: %s\n", ਰਸਤਾ);
        fprintf(stderr, "  Mode: %s\n", ਬਹੁਲਕ);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
    ਵਾਪਸ fp;
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ fopenCheck(ਰਸਤਾ, ਬਹੁਲਕ) fopen_check(ਰਸਤਾ, ਬਹੁਲਕ, __FILE__, __LINE__)

ਬਾਹਰੀ ਸਤਰਅੰਦਰ ਖਾਲੀ fread_check(ਖਾਲੀ *ptr, size_t ਮਾਪ, size_t nmemb, FILE *ਧਾਰਾ, ਅਚਲ ਅੱਖਰ *file, ਪੂਰਨ ਸਤਰ) {
    size_t ਨਤੀਜਾ = fread(ptr, ਮਾਪ, nmemb, ਧਾਰਾ);
    ਜੇ (ਨਤੀਜਾ != nmemb) {
        ਜੇ (feof(ਧਾਰਾ)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, ਸਤਰ);
        } ਨਹੀਂਤਾਂ ਜੇ (ferror(ਧਾਰਾ)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, ਸਤਰ);
        } ਨਹੀਂਤਾਂ {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, ਸਤਰ, nmemb, ਨਤੀਜਾ);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ਸਤਰ);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", ਨਤੀਜਾ);
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ freadCheck(ptr, ਮਾਪ, nmemb, ਧਾਰਾ) fread_check(ptr, ਮਾਪ, nmemb, ਧਾਰਾ, __FILE__, __LINE__)

ਬਾਹਰੀ ਸਤਰਅੰਦਰ ਖਾਲੀ fclose_check(FILE *fp, ਅਚਲ ਅੱਖਰ *file, ਪੂਰਨ ਸਤਰ) {
    ਜੇ (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, ਸਤਰ);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ਸਤਰ);
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

ਬਾਹਰੀ ਸਤਰਅੰਦਰ ਖਾਲੀ sclose_check(ਪੂਰਨ sockfd, ਅਚਲ ਅੱਖਰ *file, ਪੂਰਨ ਸਤਰ) {
    ਜੇ (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, ਸਤਰ);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ਸਤਰ);
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
ਬਾਹਰੀ ਸਤਰਅੰਦਰ ਖਾਲੀ closesocket_check(ਪੂਰਨ sockfd, ਅਚਲ ਅੱਖਰ *file, ਪੂਰਨ ਸਤਰ) {
    ਜੇ (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, ਸਤਰ);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ਸਤਰ);
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#ਜੇ_ਅੰਤ

ਬਾਹਰੀ ਸਤਰਅੰਦਰ ਖਾਲੀ fseek_check(FILE *fp, ਲੰਮਾ off, ਪੂਰਨ whence, ਅਚਲ ਅੱਖਰ *file, ਪੂਰਨ ਸਤਰ) {
    ਜੇ (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, ਸਤਰ);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", ਸਤਰ);
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

ਬਾਹਰੀ ਸਤਰਅੰਦਰ ਖਾਲੀ fwrite_check(ਖਾਲੀ *ptr, size_t ਮਾਪ, size_t nmemb, FILE *ਧਾਰਾ, ਅਚਲ ਅੱਖਰ *file, ਪੂਰਨ ਸਤਰ) {
    size_t ਨਤੀਜਾ = fwrite(ptr, ਮਾਪ, nmemb, ਧਾਰਾ);
    ਜੇ (ਨਤੀਜਾ != nmemb) {
        ਜੇ (feof(ਧਾਰਾ)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, ਸਤਰ);
        } ਨਹੀਂਤਾਂ ਜੇ (ferror(ਧਾਰਾ)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, ਸਤਰ);
        } ਨਹੀਂਤਾਂ {
            fprintf(stderr, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, ਸਤਰ, nmemb, ਨਤੀਜਾ);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ਸਤਰ);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Written elements: %zu\n", ਨਤੀਜਾ);
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ fwriteCheck(ptr, ਮਾਪ, nmemb, ਧਾਰਾ) fwrite_check(ptr, ਮਾਪ, nmemb, ਧਾਰਾ, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

ਬਾਹਰੀ ਸਤਰਅੰਦਰ ਖਾਲੀ *malloc_check(size_t ਮਾਪ, ਅਚਲ ਅੱਖਰ *file, ਪੂਰਨ ਸਤਰ) {
    ਖਾਲੀ *ptr = ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਮਾਪ);
    ਜੇ (ptr == ਨਿਰਾ) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, ਸਤਰ);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ਸਤਰ);
        fprintf(stderr, "  Size: %zu bytes\n", ਮਾਪ);
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
    ਵਾਪਸ ptr;
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ mallocCheck(ਮਾਪ) malloc_check(ਮਾਪ, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
ਬਾਹਰੀ ਸਤਰਅੰਦਰ ਖਾਲੀ token_check(ਅਚਲ ਪੂਰਨ* ਟੋਕਨ, ਪੂਰਨ token_ਗਿਣਤੀ, ਪੂਰਨ vocab_ਮਾਪ, ਅਚਲ ਅੱਖਰ *file, ਪੂਰਨ ਸਤਰ) {
    ਲਈ(ਪੂਰਨ i = 0; i < token_ਗਿਣਤੀ; i++) {
        ਜੇ(!(0 <= ਟੋਕਨ[i] && ਟੋਕਨ[i] < vocab_ਮਾਪ)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, ਸਤਰ);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", ਸਤਰ);
            fprintf(stderr, "  Token: %d\n", ਟੋਕਨ[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_ਮਾਪ);
            ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
        }
    }
}
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ tokenCheck(ਟੋਕਨ, ਗਿਣਤੀ_2, vocab) token_check(ਟੋਕਨ, ਗਿਣਤੀ_2, vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

ਬਾਹਰੀ ਸਤਰਅੰਦਰ ਖਾਲੀ create_dir_if_not_exists(ਅਚਲ ਅੱਖਰ *dir) {
    ਜੇ (dir == ਨਿਰਾ) { ਵਾਪਸ; }
    ਬਣਤਰ stat st = {0};
    ਜੇ (stat(dir, &st) == -1) {
        ਜੇ (mkdir(dir, 0700) == -1) {
            ਲਿਖੋ("ERROR: could not create directory: %s\n", dir);
            ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
        }
        ਲਿਖੋ("created directory: %s\n", dir);
    }
}

ਬਾਹਰੀ ਸਤਰਅੰਦਰ ਪੂਰਨ ਲੱਭੋ_max_ਕਦਮ(ਅਚਲ ਅੱਖਰ* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    ਜੇ (output_log_dir == ਨਿਰਾ) { ਵਾਪਸ -1; }
    DIR* dir;
    ਬਣਤਰ dirent* ਇੰਦਰਾਜ;
    ਪੂਰਨ max_ਕਦਮ = -1;
    dir = opendir(output_log_dir);
    ਜੇ (dir == ਨਿਰਾ) { ਵਾਪਸ -1; }
    ਜਦੋਂ ((ਇੰਦਰਾਜ = readdir(dir)) != ਨਿਰਾ) {
        ਜੇ (ਲੜੀ_ਤੁਲਨਾ(ਇੰਦਰਾਜ->d_name, "DONE_", 5) == 0) {
            ਪੂਰਨ ਕਦਮ = atoi(ਇੰਦਰਾਜ->d_name + 5);
            ਜੇ (ਕਦਮ > max_ਕਦਮ) {
                max_ਕਦਮ = ਕਦਮ;
            }
        }
    }
    closedir(dir);
    ਵਾਪਸ max_ਕਦਮ;
}

ਬਾਹਰੀ ਸਤਰਅੰਦਰ ਪੂਰਨ ਸਿਰੇ_with_bin(ਅਚਲ ਅੱਖਰ* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    ਜੇ (str == ਨਿਰਾ) { ਵਾਪਸ 0; }
    size_t len = ਲੜੀ_ਲੰਬਾਈ(str);
    ਅਚਲ ਅੱਖਰ* suffix = ".bin";
    size_t suffix_len = ਲੜੀ_ਲੰਬਾਈ(suffix);
    ਜੇ (len < suffix_len) { ਵਾਪਸ 0; }
    ਪੂਰਨ suffix_matches = ਲੜੀ_ਤੁਲਨਾ(str + len - suffix_len, suffix, suffix_len) == 0;
    ਵਾਪਸ suffix_matches;
}

#ਜੇ_ਅੰਤ