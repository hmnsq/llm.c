/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#اگر_تعریف_نشده UTILS_H
#تعریف_کن UTILS_H

#ضمن <unistd.h>
#ضمن <string.h>
#ضمن <stdio.h>
#ضمن <stdlib.h>
#ضمن <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#اگر_تعریف_نشده _WIN32
#ضمن <dirent.h>
#ضمن <arpa/inet.h>
#پایان_اگر

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

خارجي مضمن FILE *fopen_check(ثابت حرف *مسیر, ثابت حرف *نما, ثابت حرف *file, صحيح سطر) {
    FILE *fp = fopen(مسیر, نما);
    إذا (fp == عدم) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", مسیر, file, سطر);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سطر);
        fprintf(stderr, "  Path: %s\n", مسیر);
        fprintf(stderr, "  Mode: %s\n", نما);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        خروج(EXIT_FAILURE);
    }
    أرجع fp;
}

#تعریف_کن fopenCheck(مسیر, نما) fopen_check(مسیر, نما, __FILE__, __LINE__)

خارجي مضمن فراغ fread_check(فراغ *ptr, size_t اندازه, size_t nmemb, FILE *جریان, ثابت حرف *file, صحيح سطر) {
    size_t نتیجه = fread(ptr, اندازه, nmemb, جریان);
    إذا (نتیجه != nmemb) {
        إذا (feof(جریان)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, سطر);
        } وإلا إذا (ferror(جریان)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, سطر);
        } وإلا {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, سطر, nmemb, نتیجه);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سطر);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", نتیجه);
        خروج(EXIT_FAILURE);
    }
}

#تعریف_کن freadCheck(ptr, اندازه, nmemb, جریان) fread_check(ptr, اندازه, nmemb, جریان, __FILE__, __LINE__)

خارجي مضمن فراغ fclose_check(FILE *fp, ثابت حرف *file, صحيح سطر) {
    إذا (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, سطر);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سطر);
        خروج(EXIT_FAILURE);
    }
}

#تعریف_کن fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

خارجي مضمن فراغ sclose_check(صحيح sockfd, ثابت حرف *file, صحيح سطر) {
    إذا (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, سطر);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سطر);
        خروج(EXIT_FAILURE);
    }
}

#تعریف_کن scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#اگر_تعریف_شده _WIN32
خارجي مضمن فراغ closesocket_check(صحيح sockfd, ثابت حرف *file, صحيح سطر) {
    إذا (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, سطر);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سطر);
        خروج(EXIT_FAILURE);
    }
}

#تعریف_کن closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#پایان_اگر

خارجي مضمن فراغ fseek_check(FILE *fp, دراز off, صحيح whence, ثابت حرف *file, صحيح سطر) {
    إذا (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, سطر);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", سطر);
        خروج(EXIT_FAILURE);
    }
}

#تعریف_کن fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

خارجي مضمن فراغ fwrite_check(فراغ *ptr, size_t اندازه, size_t nmemb, FILE *جریان, ثابت حرف *file, صحيح سطر) {
    size_t نتیجه = fwrite(ptr, اندازه, nmemb, جریان);
    إذا (نتیجه != nmemb) {
        إذا (feof(جریان)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, سطر);
        } وإلا إذا (ferror(جریان)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, سطر);
        } وإلا {
            fprintf(stderr, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, سطر, nmemb, نتیجه);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سطر);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Written elements: %zu\n", نتیجه);
        خروج(EXIT_FAILURE);
    }
}

#تعریف_کن fwriteCheck(ptr, اندازه, nmemb, جریان) fwrite_check(ptr, اندازه, nmemb, جریان, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

خارجي مضمن فراغ *malloc_check(size_t اندازه, ثابت حرف *file, صحيح سطر) {
    فراغ *ptr = حافظه_بده(اندازه);
    إذا (ptr == عدم) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, سطر);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سطر);
        fprintf(stderr, "  Size: %zu bytes\n", اندازه);
        خروج(EXIT_FAILURE);
    }
    أرجع ptr;
}

#تعریف_کن mallocCheck(اندازه) malloc_check(اندازه, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
خارجي مضمن فراغ token_check(ثابت صحيح* نشانهها, صحيح token_شمار, صحيح vocab_اندازه, ثابت حرف *file, صحيح سطر) {
    لكل(صحيح i = 0; i < token_شمار; i++) {
        إذا(!(0 <= نشانهها[i] && نشانهها[i] < vocab_اندازه)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, سطر);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", سطر);
            fprintf(stderr, "  Token: %d\n", نشانهها[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_اندازه);
            خروج(EXIT_FAILURE);
        }
    }
}
#تعریف_کن tokenCheck(نشانهها, شمار, vocab) token_check(نشانهها, شمار, vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

خارجي مضمن فراغ create_dir_if_not_exists(ثابت حرف *dir) {
    إذا (dir == عدم) { أرجع; }
    بنية stat st = {0};
    إذا (stat(dir, &st) == -1) {
        إذا (mkdir(dir, 0700) == -1) {
            چاپ_کن("ERROR: could not create directory: %s\n", dir);
            خروج(EXIT_FAILURE);
        }
        چاپ_کن("created directory: %s\n", dir);
    }
}

خارجي مضمن صحيح پیدا_کن_max_گام(ثابت حرف* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    إذا (output_log_dir == عدم) { أرجع -1; }
    DIR* dir;
    بنية dirent* مدخل;
    صحيح max_گام = -1;
    dir = opendir(output_log_dir);
    إذا (dir == عدم) { أرجع -1; }
    طالما ((مدخل = readdir(dir)) != عدم) {
        إذا (مقایسه_رشته(مدخل->d_name, "DONE_", 5) == 0) {
            صحيح گام = atoi(مدخل->d_name + 5);
            إذا (گام > max_گام) {
                max_گام = گام;
            }
        }
    }
    closedir(dir);
    أرجع max_گام;
}

خارجي مضمن صحيح دوسر_with_bin(ثابت حرف* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    إذا (str == عدم) { أرجع 0; }
    size_t len = طول_رشته(str);
    ثابت حرف* suffix = ".bin";
    size_t suffix_len = طول_رشته(suffix);
    إذا (len < suffix_len) { أرجع 0; }
    صحيح suffix_matches = مقایسه_رشته(str + len - suffix_len, suffix, suffix_len) == 0;
    أرجع suffix_matches;
}

#پایان_اگر