/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#اگر_غیر_تعریف_شدہ UTILS_H
#تعریف_کرو UTILS_H

#شامل <unistd.h>
#شامل <string.h>
#شامل <stdio.h>
#شامل <stdlib.h>
#شامل <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#اگر_غیر_تعریف_شدہ _WIN32
#شامل <dirent.h>
#شامل <arpa/inet.h>
#اگر_ختم

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

خارجی اندرونی FILE *fopen_check(ثابت حرف *راستہ, ثابت حرف *منوال, ثابت حرف *file, صحیح سطر) {
    FILE *fp = fopen(راستہ, منوال);
    اگر (fp == تہی) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", راستہ, file, سطر);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سطر);
        fprintf(stderr, "  Path: %s\n", راستہ);
        fprintf(stderr, "  Mode: %s\n", منوال);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        باہر_نکلو(EXIT_FAILURE);
    }
    واپس fp;
}

#تعریف_کرو fopenCheck(راستہ, منوال) fopen_check(راستہ, منوال, __FILE__, __LINE__)

خارجی اندرونی خالی fread_check(خالی *ptr, size_t سائز, size_t nmemb, FILE *دھارا, ثابت حرف *file, صحیح سطر) {
    size_t نتیجہ = fread(ptr, سائز, nmemb, دھارا);
    اگر (نتیجہ != nmemb) {
        اگر (feof(دھارا)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, سطر);
        } ورنہ اگر (ferror(دھارا)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, سطر);
        } ورنہ {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, سطر, nmemb, نتیجہ);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سطر);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", نتیجہ);
        باہر_نکلو(EXIT_FAILURE);
    }
}

#تعریف_کرو freadCheck(ptr, سائز, nmemb, دھارا) fread_check(ptr, سائز, nmemb, دھارا, __FILE__, __LINE__)

خارجی اندرونی خالی fclose_check(FILE *fp, ثابت حرف *file, صحیح سطر) {
    اگر (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, سطر);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سطر);
        باہر_نکلو(EXIT_FAILURE);
    }
}

#تعریف_کرو fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

خارجی اندرونی خالی sclose_check(صحیح sockfd, ثابت حرف *file, صحیح سطر) {
    اگر (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, سطر);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سطر);
        باہر_نکلو(EXIT_FAILURE);
    }
}

#تعریف_کرو scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#اگر_تعریف_شدہ _WIN32
خارجی اندرونی خالی closesocket_check(صحیح sockfd, ثابت حرف *file, صحیح سطر) {
    اگر (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, سطر);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سطر);
        باہر_نکلو(EXIT_FAILURE);
    }
}

#تعریف_کرو closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#اگر_ختم

خارجی اندرونی خالی fseek_check(FILE *fp, طویل off, صحیح whence, ثابت حرف *file, صحیح سطر) {
    اگر (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, سطر);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", سطر);
        باہر_نکلو(EXIT_FAILURE);
    }
}

#تعریف_کرو fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

خارجی اندرونی خالی fwrite_check(خالی *ptr, size_t سائز, size_t nmemb, FILE *دھارا, ثابت حرف *file, صحیح سطر) {
    size_t نتیجہ = fwrite(ptr, سائز, nmemb, دھارا);
    اگر (نتیجہ != nmemb) {
        اگر (feof(دھارا)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, سطر);
        } ورنہ اگر (ferror(دھارا)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, سطر);
        } ورنہ {
            fprintf(stderr, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, سطر, nmemb, نتیجہ);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سطر);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Written elements: %zu\n", نتیجہ);
        باہر_نکلو(EXIT_FAILURE);
    }
}

#تعریف_کرو fwriteCheck(ptr, سائز, nmemb, دھارا) fwrite_check(ptr, سائز, nmemb, دھارا, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

خارجی اندرونی خالی *malloc_check(size_t سائز, ثابت حرف *file, صحیح سطر) {
    خالی *ptr = حافظہ_دو(سائز);
    اگر (ptr == تہی) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, سطر);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سطر);
        fprintf(stderr, "  Size: %zu bytes\n", سائز);
        باہر_نکلو(EXIT_FAILURE);
    }
    واپس ptr;
}

#تعریف_کرو mallocCheck(سائز) malloc_check(سائز, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
خارجی اندرونی خالی token_check(ثابت صحیح* علامات, صحیح token_گنتی, صحیح vocab_سائز, ثابت حرف *file, صحیح سطر) {
    برائے(صحیح i = 0; i < token_گنتی; i++) {
        اگر(!(0 <= علامات[i] && علامات[i] < vocab_سائز)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, سطر);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", سطر);
            fprintf(stderr, "  Token: %d\n", علامات[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_سائز);
            باہر_نکلو(EXIT_FAILURE);
        }
    }
}
#تعریف_کرو tokenCheck(علامات, گنتی, vocab) token_check(علامات, گنتی, vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

خارجی اندرونی خالی create_dir_if_not_exists(ثابت حرف *dir) {
    اگر (dir == تہی) { واپس; }
    ساخت stat st = {0};
    اگر (stat(dir, &st) == -1) {
        اگر (mkdir(dir, 0700) == -1) {
            لکھو("ERROR: could not create directory: %s\n", dir);
            باہر_نکلو(EXIT_FAILURE);
        }
        لکھو("created directory: %s\n", dir);
    }
}

خارجی اندرونی صحیح ڈھونڈو_max_قدم(ثابت حرف* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    اگر (output_log_dir == تہی) { واپس -1; }
    DIR* dir;
    ساخت dirent* اندراج_2;
    صحیح max_قدم = -1;
    dir = opendir(output_log_dir);
    اگر (dir == تہی) { واپس -1; }
    جب ((اندراج_2 = readdir(dir)) != تہی) {
        اگر (لڑی_موازنہ(اندراج_2->d_name, "DONE_", 5) == 0) {
            صحیح قدم = atoi(اندراج_2->d_name + 5);
            اگر (قدم > max_قدم) {
                max_قدم = قدم;
            }
        }
    }
    closedir(dir);
    واپس max_قدم;
}

خارجی اندرونی صحیح سرے_with_bin(ثابت حرف* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    اگر (str == تہی) { واپس 0; }
    size_t len = لڑی_لمبائی(str);
    ثابت حرف* suffix = ".bin";
    size_t suffix_len = لڑی_لمبائی(suffix);
    اگر (len < suffix_len) { واپس 0; }
    صحیح suffix_matches = لڑی_موازنہ(str + len - suffix_len, suffix, suffix_len) == 0;
    واپس suffix_matches;
}

#اگر_ختم