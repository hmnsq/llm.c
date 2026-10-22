/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#جے_بنا_تعریف UTILS_H
#تعریف_کرو UTILS_H

#ضمن <unistd.h>
#ضمن <string.h>
#ضمن <stdio.h>
#ضمن <stdlib.h>
#ضمن <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#جے_بنا_تعریف _WIN32
#ضمن <dirent.h>
#ضمن <arpa/inet.h>
#جے_انت

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

خارجي مضمن FILE *fopen_check(ثابت حرف *رستہ, ثابت حرف *منوال, ثابت حرف *file, صحيح سطر) {
    FILE *fp = fopen(رستہ, منوال);
    إذا (fp == عدم) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", رستہ, file, سطر);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سطر);
        fprintf(stderr, "  Path: %s\n", رستہ);
        fprintf(stderr, "  Mode: %s\n", منوال);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        باہر_نکلو(EXIT_FAILURE);
    }
    أرجع fp;
}

#تعریف_کرو fopenCheck(رستہ, منوال) fopen_check(رستہ, منوال, __FILE__, __LINE__)

خارجي مضمن فراغ fread_check(فراغ *ptr, size_t ناپ, size_t nmemb, FILE *دھارا, ثابت حرف *file, صحيح سطر) {
    size_t نتیجہ = fread(ptr, ناپ, nmemb, دھارا);
    إذا (نتیجہ != nmemb) {
        إذا (feof(دھارا)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, سطر);
        } وإلا إذا (ferror(دھارا)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, سطر);
        } وإلا {
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

#تعریف_کرو freadCheck(ptr, ناپ, nmemb, دھارا) fread_check(ptr, ناپ, nmemb, دھارا, __FILE__, __LINE__)

خارجي مضمن فراغ fclose_check(FILE *fp, ثابت حرف *file, صحيح سطر) {
    إذا (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, سطر);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سطر);
        باہر_نکلو(EXIT_FAILURE);
    }
}

#تعریف_کرو fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

خارجي مضمن فراغ sclose_check(صحيح sockfd, ثابت حرف *file, صحيح سطر) {
    إذا (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, سطر);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سطر);
        باہر_نکلو(EXIT_FAILURE);
    }
}

#تعریف_کرو scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#جے_تعریف_شدہ _WIN32
خارجي مضمن فراغ closesocket_check(صحيح sockfd, ثابت حرف *file, صحيح سطر) {
    إذا (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, سطر);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سطر);
        باہر_نکلو(EXIT_FAILURE);
    }
}

#تعریف_کرو closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#جے_انت

خارجي مضمن فراغ fseek_check(FILE *fp, لما off, صحيح whence, ثابت حرف *file, صحيح سطر) {
    إذا (fseek(fp, off, whence) != 0) {
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

خارجي مضمن فراغ fwrite_check(فراغ *ptr, size_t ناپ, size_t nmemb, FILE *دھارا, ثابت حرف *file, صحيح سطر) {
    size_t نتیجہ = fwrite(ptr, ناپ, nmemb, دھارا);
    إذا (نتیجہ != nmemb) {
        إذا (feof(دھارا)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, سطر);
        } وإلا إذا (ferror(دھارا)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, سطر);
        } وإلا {
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

#تعریف_کرو fwriteCheck(ptr, ناپ, nmemb, دھارا) fwrite_check(ptr, ناپ, nmemb, دھارا, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

خارجي مضمن فراغ *malloc_check(size_t ناپ, ثابت حرف *file, صحيح سطر) {
    فراغ *ptr = حافظہ_دیو(ناپ);
    إذا (ptr == عدم) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, سطر);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سطر);
        fprintf(stderr, "  Size: %zu bytes\n", ناپ);
        باہر_نکلو(EXIT_FAILURE);
    }
    أرجع ptr;
}

#تعریف_کرو mallocCheck(ناپ) malloc_check(ناپ, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
خارجي مضمن فراغ token_check(ثابت صحيح* ٹوکن, صحيح token_گنتی, صحيح vocab_ناپ, ثابت حرف *file, صحيح سطر) {
    لكل(صحيح i = 0; i < token_گنتی; i++) {
        إذا(!(0 <= ٹوکن[i] && ٹوکن[i] < vocab_ناپ)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, سطر);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", سطر);
            fprintf(stderr, "  Token: %d\n", ٹوکن[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_ناپ);
            باہر_نکلو(EXIT_FAILURE);
        }
    }
}
#تعریف_کرو tokenCheck(ٹوکن, گنتی, vocab) token_check(ٹوکن, گنتی, vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

خارجي مضمن فراغ create_dir_if_not_exists(ثابت حرف *dir) {
    إذا (dir == عدم) { أرجع; }
    بنية stat st = {0};
    إذا (stat(dir, &st) == -1) {
        إذا (mkdir(dir, 0700) == -1) {
            لکھو("ERROR: could not create directory: %s\n", dir);
            باہر_نکلو(EXIT_FAILURE);
        }
        لکھو("created directory: %s\n", dir);
    }
}

خارجي مضمن صحيح لبھو_max_قدم(ثابت حرف* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    إذا (output_log_dir == عدم) { أرجع -1; }
    DIR* dir;
    بنية dirent* اندراج;
    صحيح max_قدم = -1;
    dir = opendir(output_log_dir);
    إذا (dir == عدم) { أرجع -1; }
    طالما ((اندراج = readdir(dir)) != عدم) {
        إذا (لڑی_مقابلہ(اندراج->d_name, "DONE_", 5) == 0) {
            صحيح قدم = atoi(اندراج->d_name + 5);
            إذا (قدم > max_قدم) {
                max_قدم = قدم;
            }
        }
    }
    closedir(dir);
    أرجع max_قدم;
}

خارجي مضمن صحيح سرے_with_bin(ثابت حرف* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    إذا (str == عدم) { أرجع 0; }
    size_t len = لڑی_لمبائی(str);
    ثابت حرف* suffix = ".bin";
    size_t suffix_len = لڑی_لمبائی(suffix);
    إذا (len < suffix_len) { أرجع 0; }
    صحيح suffix_matches = لڑی_مقابلہ(str + len - suffix_len, suffix, suffix_len) == 0;
    أرجع suffix_matches;
}

#جے_انت