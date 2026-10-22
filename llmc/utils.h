/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#جيڪڏهن_وصف_نه_ٿيل UTILS_H
#وصف_ڪريو UTILS_H

#ضمن <unistd.h>
#ضمن <string.h>
#ضمن <stdio.h>
#ضمن <stdlib.h>
#ضمن <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#جيڪڏهن_وصف_نه_ٿيل _WIN32
#ضمن <dirent.h>
#ضمن <arpa/inet.h>
#جيڪڏهن_پورو

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

خارجي مضمن FILE *fopen_check(ثابت حرف *رستو, ثابت حرف *گهڻائي, ثابت حرف *file, صحيح سٽ) {
    FILE *fp = fopen(رستو, گهڻائي);
    إذا (fp == عدم) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", رستو, file, سٽ);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سٽ);
        fprintf(stderr, "  Path: %s\n", رستو);
        fprintf(stderr, "  Mode: %s\n", گهڻائي);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        ٻاهر_نڪرو(EXIT_FAILURE);
    }
    أرجع fp;
}

#وصف_ڪريو fopenCheck(رستو, گهڻائي) fopen_check(رستو, گهڻائي, __FILE__, __LINE__)

خارجي مضمن فراغ fread_check(فراغ *ptr, size_t سائيز, size_t nmemb, FILE *وهڪرو, ثابت حرف *file, صحيح سٽ) {
    size_t نتيجو = fread(ptr, سائيز, nmemb, وهڪرو);
    إذا (نتيجو != nmemb) {
        إذا (feof(وهڪرو)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, سٽ);
        } وإلا إذا (ferror(وهڪرو)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, سٽ);
        } وإلا {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, سٽ, nmemb, نتيجو);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سٽ);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", نتيجو);
        ٻاهر_نڪرو(EXIT_FAILURE);
    }
}

#وصف_ڪريو freadCheck(ptr, سائيز, nmemb, وهڪرو) fread_check(ptr, سائيز, nmemb, وهڪرو, __FILE__, __LINE__)

خارجي مضمن فراغ fclose_check(FILE *fp, ثابت حرف *file, صحيح سٽ) {
    إذا (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, سٽ);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سٽ);
        ٻاهر_نڪرو(EXIT_FAILURE);
    }
}

#وصف_ڪريو fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

خارجي مضمن فراغ sclose_check(صحيح sockfd, ثابت حرف *file, صحيح سٽ) {
    إذا (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, سٽ);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سٽ);
        ٻاهر_نڪرو(EXIT_FAILURE);
    }
}

#وصف_ڪريو scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#جيڪڏهن_وصف_ٿيل _WIN32
خارجي مضمن فراغ closesocket_check(صحيح sockfd, ثابت حرف *file, صحيح سٽ) {
    إذا (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, سٽ);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سٽ);
        ٻاهر_نڪرو(EXIT_FAILURE);
    }
}

#وصف_ڪريو closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#جيڪڏهن_پورو

خارجي مضمن فراغ fseek_check(FILE *fp, ڊگهو off, صحيح whence, ثابت حرف *file, صحيح سٽ) {
    إذا (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, سٽ);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", سٽ);
        ٻاهر_نڪرو(EXIT_FAILURE);
    }
}

#وصف_ڪريو fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

خارجي مضمن فراغ fwrite_check(فراغ *ptr, size_t سائيز, size_t nmemb, FILE *وهڪرو, ثابت حرف *file, صحيح سٽ) {
    size_t نتيجو = fwrite(ptr, سائيز, nmemb, وهڪرو);
    إذا (نتيجو != nmemb) {
        إذا (feof(وهڪرو)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, سٽ);
        } وإلا إذا (ferror(وهڪرو)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, سٽ);
        } وإلا {
            fprintf(stderr, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, سٽ, nmemb, نتيجو);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سٽ);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Written elements: %zu\n", نتيجو);
        ٻاهر_نڪرو(EXIT_FAILURE);
    }
}

#وصف_ڪريو fwriteCheck(ptr, سائيز, nmemb, وهڪرو) fwrite_check(ptr, سائيز, nmemb, وهڪرو, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

خارجي مضمن فراغ *malloc_check(size_t سائيز, ثابت حرف *file, صحيح سٽ) {
    فراغ *ptr = يادگيري_ڏيو(سائيز);
    إذا (ptr == عدم) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, سٽ);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", سٽ);
        fprintf(stderr, "  Size: %zu bytes\n", سائيز);
        ٻاهر_نڪرو(EXIT_FAILURE);
    }
    أرجع ptr;
}

#وصف_ڪريو mallocCheck(سائيز) malloc_check(سائيز, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
خارجي مضمن فراغ token_check(ثابت صحيح* ٽوڪن, صحيح token_ڳڻپ, صحيح vocab_سائيز, ثابت حرف *file, صحيح سٽ) {
    لكل(صحيح i = 0; i < token_ڳڻپ; i++) {
        إذا(!(0 <= ٽوڪن[i] && ٽوڪن[i] < vocab_سائيز)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, سٽ);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", سٽ);
            fprintf(stderr, "  Token: %d\n", ٽوڪن[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_سائيز);
            ٻاهر_نڪرو(EXIT_FAILURE);
        }
    }
}
#وصف_ڪريو tokenCheck(ٽوڪن, ڳڻپ, vocab) token_check(ٽوڪن, ڳڻپ, vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

خارجي مضمن فراغ create_dir_if_not_exists(ثابت حرف *dir) {
    إذا (dir == عدم) { أرجع; }
    بنية stat st = {0};
    إذا (stat(dir, &st) == -1) {
        إذا (mkdir(dir, 0700) == -1) {
            لکو("ERROR: could not create directory: %s\n", dir);
            ٻاهر_نڪرو(EXIT_FAILURE);
        }
        لکو("created directory: %s\n", dir);
    }
}

خارجي مضمن صحيح ڳوليو_max_قدم(ثابت حرف* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    إذا (output_log_dir == عدم) { أرجع -1; }
    DIR* dir;
    بنية dirent* داخلا_2;
    صحيح max_قدم = -1;
    dir = opendir(output_log_dir);
    إذا (dir == عدم) { أرجع -1; }
    طالما ((داخلا_2 = readdir(dir)) != عدم) {
        إذا (لڙي_ڀيٽ(داخلا_2->d_name, "DONE_", 5) == 0) {
            صحيح قدم = atoi(داخلا_2->d_name + 5);
            إذا (قدم > max_قدم) {
                max_قدم = قدم;
            }
        }
    }
    closedir(dir);
    أرجع max_قدم;
}

خارجي مضمن صحيح پڇاڙيون_with_bin(ثابت حرف* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    إذا (str == عدم) { أرجع 0; }
    size_t len = لڙي_ڊيگهه(str);
    ثابت حرف* suffix = ".bin";
    size_t suffix_len = لڙي_ڊيگهه(suffix);
    إذا (len < suffix_len) { أرجع 0; }
    صحيح suffix_matches = لڙي_ڀيٽ(str + len - suffix_len, suffix, suffix_len) == 0;
    أرجع suffix_matches;
}

#جيڪڏهن_پورو