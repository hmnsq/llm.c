/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#که_نه_تعریف_شوی UTILS_H
#تعریف_کړه UTILS_H

#ضمن <unistd.h>
#ضمن <string.h>
#ضمن <stdio.h>
#ضمن <stdlib.h>
#ضمن <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#که_نه_تعریف_شوی _WIN32
#ضمن <dirent.h>
#ضمن <arpa/inet.h>
#د_که_پای

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

خارجي مضمن FILE *fopen_check(ثابت حرف *لار, ثابت حرف *ډېرپېښ, ثابت حرف *file, صحيح کرښه) {
    FILE *fp = fopen(لار, ډېرپېښ);
    إذا (fp == عدم) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", لار, file, کرښه);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", کرښه);
        fprintf(stderr, "  Path: %s\n", لار);
        fprintf(stderr, "  Mode: %s\n", ډېرپېښ);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        وتل(EXIT_FAILURE);
    }
    أرجع fp;
}

#تعریف_کړه fopenCheck(لار, ډېرپېښ) fopen_check(لار, ډېرپېښ, __FILE__, __LINE__)

خارجي مضمن فراغ fread_check(فراغ *ptr, size_t اندازه, size_t nmemb, FILE *بهیر, ثابت حرف *file, صحيح کرښه) {
    size_t پایله = fread(ptr, اندازه, nmemb, بهیر);
    إذا (پایله != nmemb) {
        إذا (feof(بهیر)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, کرښه);
        } وإلا إذا (ferror(بهیر)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, کرښه);
        } وإلا {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, کرښه, nmemb, پایله);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", کرښه);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", پایله);
        وتل(EXIT_FAILURE);
    }
}

#تعریف_کړه freadCheck(ptr, اندازه, nmemb, بهیر) fread_check(ptr, اندازه, nmemb, بهیر, __FILE__, __LINE__)

خارجي مضمن فراغ fclose_check(FILE *fp, ثابت حرف *file, صحيح کرښه) {
    إذا (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, کرښه);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", کرښه);
        وتل(EXIT_FAILURE);
    }
}

#تعریف_کړه fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

خارجي مضمن فراغ sclose_check(صحيح sockfd, ثابت حرف *file, صحيح کرښه) {
    إذا (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, کرښه);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", کرښه);
        وتل(EXIT_FAILURE);
    }
}

#تعریف_کړه scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#که_تعریف_شوی _WIN32
خارجي مضمن فراغ closesocket_check(صحيح sockfd, ثابت حرف *file, صحيح کرښه) {
    إذا (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, کرښه);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", کرښه);
        وتل(EXIT_FAILURE);
    }
}

#تعریف_کړه closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#د_که_پای

خارجي مضمن فراغ fseek_check(FILE *fp, اوږد off, صحيح whence, ثابت حرف *file, صحيح کرښه) {
    إذا (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, کرښه);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", کرښه);
        وتل(EXIT_FAILURE);
    }
}

#تعریف_کړه fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

خارجي مضمن فراغ fwrite_check(فراغ *ptr, size_t اندازه, size_t nmemb, FILE *بهیر, ثابت حرف *file, صحيح کرښه) {
    size_t پایله = fwrite(ptr, اندازه, nmemb, بهیر);
    إذا (پایله != nmemb) {
        إذا (feof(بهیر)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, کرښه);
        } وإلا إذا (ferror(بهیر)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, کرښه);
        } وإلا {
            fprintf(stderr, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, کرښه, nmemb, پایله);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", کرښه);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Written elements: %zu\n", پایله);
        وتل(EXIT_FAILURE);
    }
}

#تعریف_کړه fwriteCheck(ptr, اندازه, nmemb, بهیر) fwrite_check(ptr, اندازه, nmemb, بهیر, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

خارجي مضمن فراغ *malloc_check(size_t اندازه, ثابت حرف *file, صحيح کرښه) {
    فراغ *ptr = حافظه_ورکړه(اندازه);
    إذا (ptr == عدم) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, کرښه);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", کرښه);
        fprintf(stderr, "  Size: %zu bytes\n", اندازه);
        وتل(EXIT_FAILURE);
    }
    أرجع ptr;
}

#تعریف_کړه mallocCheck(اندازه) malloc_check(اندازه, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
خارجي مضمن فراغ token_check(ثابت صحيح* نښې, صحيح token_شمېر, صحيح vocab_اندازه, ثابت حرف *file, صحيح کرښه) {
    لكل(صحيح i = 0; i < token_شمېر; i++) {
        إذا(!(0 <= نښې[i] && نښې[i] < vocab_اندازه)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, کرښه);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", کرښه);
            fprintf(stderr, "  Token: %d\n", نښې[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_اندازه);
            وتل(EXIT_FAILURE);
        }
    }
}
#تعریف_کړه tokenCheck(نښې, شمېر, vocab) token_check(نښې, شمېر, vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

خارجي مضمن فراغ create_dir_if_not_exists(ثابت حرف *dir) {
    إذا (dir == عدم) { أرجع; }
    بنية stat st = {0};
    إذا (stat(dir, &st) == -1) {
        إذا (mkdir(dir, 0700) == -1) {
            چاپ_کړه("ERROR: could not create directory: %s\n", dir);
            وتل(EXIT_FAILURE);
        }
        چاپ_کړه("created directory: %s\n", dir);
    }
}

خارجي مضمن صحيح پیدا_کړه_max_ګام(ثابت حرف* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    إذا (output_log_dir == عدم) { أرجع -1; }
    DIR* dir;
    بنية dirent* ثبت;
    صحيح max_ګام = -1;
    dir = opendir(output_log_dir);
    إذا (dir == عدم) { أرجع -1; }
    طالما ((ثبت = readdir(dir)) != عدم) {
        إذا (مزي_پرتله(ثبت->d_name, "DONE_", 5) == 0) {
            صحيح ګام = atoi(ثبت->d_name + 5);
            إذا (ګام > max_ګام) {
                max_ګام = ګام;
            }
        }
    }
    closedir(dir);
    أرجع max_ګام;
}

خارجي مضمن صحيح پایونه_with_bin(ثابت حرف* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    إذا (str == عدم) { أرجع 0; }
    size_t len = مزي_اوږدوالی(str);
    ثابت حرف* suffix = ".bin";
    size_t suffix_len = مزي_اوږدوالی(suffix);
    إذا (len < suffix_len) { أرجع 0; }
    صحيح suffix_matches = مزي_پرتله(str + len - suffix_len, suffix, suffix_len) == 0;
    أرجع suffix_matches;
}

#د_که_پای