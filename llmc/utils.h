/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#ifndef UTILS_H
#define UTILS_H

#அந்தர்பாவ <unistd.h>
#அந்தர்பாவ <string.h>
#அந்தர்பாவ <stdio.h>
#அந்தர்பாவ <stdlib.h>
#அந்தர்பாவ <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#ifndef _WIN32
#அந்தர்பாவ <dirent.h>
#அந்தர்பாவ <arpa/inet.h>
#endif

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

வெளி உள்வரி FILE *fopen_check(மாறிலி எழுத்து *path, மாறிலி எழுத்து *mode, மாறிலி எழுத்து *file, முழு வரி) {
    FILE *fp = fopen(path, mode);
    எனில் (fp == வெற்று) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, வரி);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", வரி);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    திரும்பு fp;
}

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

வெளி உள்வரி வெறுமை fread_check(வெறுமை *ptr, size_t size, size_t nmemb, FILE *stream, மாறிலி எழுத்து *file, முழு வரி) {
    size_t result = fread(ptr, size, nmemb, stream);
    எனில் (result != nmemb) {
        எனில் (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, வரி);
        } இல்லையெனில் எனில் (ferror(stream)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, வரி);
        } இல்லையெனில் {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, வரி, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", வரி);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

வெளி உள்வரி வெறுமை fclose_check(FILE *fp, மாறிலி எழுத்து *file, முழு வரி) {
    எனில் (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, வரி);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", வரி);
        exit(EXIT_FAILURE);
    }
}

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

வெளி உள்வரி வெறுமை sclose_check(முழு sockfd, மாறிலி எழுத்து *file, முழு வரி) {
    எனில் (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, வரி);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", வரி);
        exit(EXIT_FAILURE);
    }
}

#define scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#ifdef _WIN32
வெளி உள்வரி வெறுமை closesocket_check(முழு sockfd, மாறிலி எழுத்து *file, முழு வரி) {
    எனில் (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, வரி);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", வரி);
        exit(EXIT_FAILURE);
    }
}

#define closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#endif

வெளி உள்வரி வெறுமை fseek_check(FILE *fp, long off, முழு whence, மாறிலி எழுத்து *file, முழு வரி) {
    எனில் (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, வரி);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", வரி);
        exit(EXIT_FAILURE);
    }
}

#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

வெளி உள்வரி வெறுமை fwrite_check(வெறுமை *ptr, size_t size, size_t nmemb, FILE *stream, மாறிலி எழுத்து *file, முழு வரி) {
    size_t result = fwrite(ptr, size, nmemb, stream);
    எனில் (result != nmemb) {
        எனில் (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, வரி);
        } இல்லையெனில் எனில் (ferror(stream)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, வரி);
        } இல்லையெனில் {
            fprintf(stderr, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, வரி, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", வரி);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Written elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define fwriteCheck(ptr, size, nmemb, stream) fwrite_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

வெளி உள்வரி வெறுமை *malloc_check(size_t size, மாறிலி எழுத்து *file, முழு வரி) {
    வெறுமை *ptr = malloc(size);
    எனில் (ptr == வெற்று) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, வரி);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", வரி);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    திரும்பு ptr;
}

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
வெளி உள்வரி வெறுமை token_check(மாறிலி முழு* tokens, முழு token_மொத்தம், முழு vocab_size, மாறிலி எழுத்து *file, முழு வரி) {
    க்கு(முழு i = 0; i < token_மொத்தம்; i++) {
        எனில்(!(0 <= tokens[i] && tokens[i] < vocab_size)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, வரி);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", வரி);
            fprintf(stderr, "  Token: %d\n", tokens[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_size);
            exit(EXIT_FAILURE);
        }
    }
}
#define tokenCheck(tokens, மொத்தம், vocab) token_check(tokens, மொத்தம், vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

வெளி உள்வரி வெறுமை create_dir_if_not_exists(மாறிலி எழுத்து *dir) {
    எனில் (dir == வெற்று) { திரும்பு; }
    கட்டமைப்பு stat st = {0};
    எனில் (stat(dir, &st) == -1) {
        எனில் (mkdir(dir, 0700) == -1) {
            printf("ERROR: could not create directory: %s\n", dir);
            exit(EXIT_FAILURE);
        }
        printf("created directory: %s\n", dir);
    }
}

வெளி உள்வரி முழு find_max_step(மாறிலி எழுத்து* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    எனில் (output_log_dir == வெற்று) { திரும்பு -1; }
    DIR* dir;
    கட்டமைப்பு dirent* entry;
    முழு max_step = -1;
    dir = opendir(output_log_dir);
    எனில் (dir == வெற்று) { திரும்பு -1; }
    போது ((entry = readdir(dir)) != வெற்று) {
        எனில் (strncmp(entry->d_name, "DONE_", 5) == 0) {
            முழு step = atoi(entry->d_name + 5);
            எனில் (step > max_step) {
                max_step = step;
            }
        }
    }
    closedir(dir);
    திரும்பு max_step;
}

வெளி உள்வரி முழு ends_with_bin(மாறிலி எழுத்து* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    எனில் (str == வெற்று) { திரும்பு 0; }
    size_t len = strlen(str);
    மாறிலி எழுத்து* suffix = ".bin";
    size_t suffix_len = strlen(suffix);
    எனில் (len < suffix_len) { திரும்பு 0; }
    முழு suffix_matches = strncmp(str + len - suffix_len, suffix, suffix_len) == 0;
    திரும்பு suffix_matches;
}

#endif