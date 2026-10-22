/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ UTILS_H
#ವ್ಯಾಖ್ಯಾನಿಸು UTILS_H

#ಅನ್ತರ್ಭಾವ <unistd.h>
#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ _WIN32
#ಅನ್ತರ್ಭಾವ <dirent.h>
#ಅನ್ತರ್ಭಾವ <arpa/inet.h>
#ಆದರೆ_ಅಂತ್ಯ

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

ಬಾಹ್ಯ ಸಾಲೊಳಗೆ FILE *fopen_check(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *ಮಾರ್ಗ, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *ಬಹುಲಕ, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *file, ಪೂರ್ಣಾಂಕ ಸಾಲು) {
    FILE *fp = fopen(ಮಾರ್ಗ, ಬಹುಲಕ);
    ಆದರೆ (fp == ಖಾಲಿ) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", ಮಾರ್ಗ, file, ಸಾಲು);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ಸಾಲು);
        fprintf(stderr, "  Path: %s\n", ಮಾರ್ಗ);
        fprintf(stderr, "  Mode: %s\n", ಬಹುಲಕ);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
    ಹಿಂತಿರುಗಿ fp;
}

#ವ್ಯಾಖ್ಯಾನಿಸು fopenCheck(ಮಾರ್ಗ, ಬಹುಲಕ) fopen_check(ಮಾರ್ಗ, ಬಹುಲಕ, __FILE__, __LINE__)

ಬಾಹ್ಯ ಸಾಲೊಳಗೆ ಶೂನ್ಯ fread_check(ಶೂನ್ಯ *ptr, size_t ಗಾತ್ರ_2, size_t nmemb, FILE *ಹರಿವು, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *file, ಪೂರ್ಣಾಂಕ ಸಾಲು) {
    size_t ಫಲಿತಾಂಶ = fread(ptr, ಗಾತ್ರ_2, nmemb, ಹರಿವು);
    ಆದರೆ (ಫಲಿತಾಂಶ != nmemb) {
        ಆದರೆ (feof(ಹರಿವು)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, ಸಾಲು);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ferror(ಹರಿವು)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, ಸಾಲು);
        } ಇಲ್ಲದಿದ್ದರೆ {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, ಸಾಲು, nmemb, ಫಲಿತಾಂಶ);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ಸಾಲು);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", ಫಲಿತಾಂಶ);
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು freadCheck(ptr, ಗಾತ್ರ_2, nmemb, ಹರಿವು) fread_check(ptr, ಗಾತ್ರ_2, nmemb, ಹರಿವು, __FILE__, __LINE__)

ಬಾಹ್ಯ ಸಾಲೊಳಗೆ ಶೂನ್ಯ fclose_check(FILE *fp, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *file, ಪೂರ್ಣಾಂಕ ಸಾಲು) {
    ಆದರೆ (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, ಸಾಲು);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ಸಾಲು);
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

ಬಾಹ್ಯ ಸಾಲೊಳಗೆ ಶೂನ್ಯ sclose_check(ಪೂರ್ಣಾಂಕ sockfd, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *file, ಪೂರ್ಣಾಂಕ ಸಾಲು) {
    ಆದರೆ (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, ಸಾಲು);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ಸಾಲು);
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
ಬಾಹ್ಯ ಸಾಲೊಳಗೆ ಶೂನ್ಯ closesocket_check(ಪೂರ್ಣಾಂಕ sockfd, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *file, ಪೂರ್ಣಾಂಕ ಸಾಲು) {
    ಆದರೆ (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, ಸಾಲು);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ಸಾಲು);
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#ಆದರೆ_ಅಂತ್ಯ

ಬಾಹ್ಯ ಸಾಲೊಳಗೆ ಶೂನ್ಯ fseek_check(FILE *fp, ದೀರ್ಘ off, ಪೂರ್ಣಾಂಕ whence, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *file, ಪೂರ್ಣಾಂಕ ಸಾಲು) {
    ಆದರೆ (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, ಸಾಲು);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", ಸಾಲು);
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

ಬಾಹ್ಯ ಸಾಲೊಳಗೆ ಶೂನ್ಯ fwrite_check(ಶೂನ್ಯ *ptr, size_t ಗಾತ್ರ_2, size_t nmemb, FILE *ಹರಿವು, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *file, ಪೂರ್ಣಾಂಕ ಸಾಲು) {
    size_t ಫಲಿತಾಂಶ = fwrite(ptr, ಗಾತ್ರ_2, nmemb, ಹರಿವು);
    ಆದರೆ (ಫಲಿತಾಂಶ != nmemb) {
        ಆದರೆ (feof(ಹರಿವು)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, ಸಾಲು);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ferror(ಹರಿವು)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, ಸಾಲು);
        } ಇಲ್ಲದಿದ್ದರೆ {
            fprintf(stderr, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, ಸಾಲು, nmemb, ಫಲಿತಾಂಶ);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ಸಾಲು);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Written elements: %zu\n", ಫಲಿತಾಂಶ);
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು fwriteCheck(ptr, ಗಾತ್ರ_2, nmemb, ಹರಿವು) fwrite_check(ptr, ಗಾತ್ರ_2, nmemb, ಹರಿವು, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

ಬಾಹ್ಯ ಸಾಲೊಳಗೆ ಶೂನ್ಯ *malloc_check(size_t ಗಾತ್ರ_2, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *file, ಪೂರ್ಣಾಂಕ ಸಾಲು) {
    ಶೂನ್ಯ *ptr = ಸ್ಮರಣೆ_ಕೊಡು(ಗಾತ್ರ_2);
    ಆದರೆ (ptr == ಖಾಲಿ) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, ಸಾಲು);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", ಸಾಲು);
        fprintf(stderr, "  Size: %zu bytes\n", ಗಾತ್ರ_2);
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
    ಹಿಂತಿರುಗಿ ptr;
}

#ವ್ಯಾಖ್ಯಾನಿಸು mallocCheck(ಗಾತ್ರ_2) malloc_check(ಗಾತ್ರ_2, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
ಬಾಹ್ಯ ಸಾಲೊಳಗೆ ಶೂನ್ಯ token_check(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ* ಟೋಕನುಗಳು, ಪೂರ್ಣಾಂಕ token_ಎಣಿಕೆ, ಪೂರ್ಣಾಂಕ vocab_ಗಾತ್ರ, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *file, ಪೂರ್ಣಾಂಕ ಸಾಲು) {
    ಗಾಗಿ(ಪೂರ್ಣಾಂಕ i = 0; i < token_ಎಣಿಕೆ; i++) {
        ಆದರೆ(!(0 <= ಟೋಕನುಗಳು[i] && ಟೋಕನುಗಳು[i] < vocab_ಗಾತ್ರ)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, ಸಾಲು);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", ಸಾಲು);
            fprintf(stderr, "  Token: %d\n", ಟೋಕನುಗಳು[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_ಗಾತ್ರ);
            ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
        }
    }
}
#ವ್ಯಾಖ್ಯಾನಿಸು tokenCheck(ಟೋಕನುಗಳು, ಎಣಿಕೆ_2, vocab) token_check(ಟೋಕನುಗಳು, ಎಣಿಕೆ_2, vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

ಬಾಹ್ಯ ಸಾಲೊಳಗೆ ಶೂನ್ಯ create_dir_if_not_exists(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *dir) {
    ಆದರೆ (dir == ಖಾಲಿ) { ಹಿಂತಿರುಗಿ; }
    ರಚನೆ stat st = {0};
    ಆದರೆ (stat(dir, &st) == -1) {
        ಆದರೆ (mkdir(dir, 0700) == -1) {
            ಬರೆ("ERROR: could not create directory: %s\n", dir);
            ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
        }
        ಬರೆ("created directory: %s\n", dir);
    }
}

ಬಾಹ್ಯ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ ಕಂಡುಹಿಡಿ_max_ಹೆಜ್ಜೆ(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    ಆದರೆ (output_log_dir == ಖಾಲಿ) { ಹಿಂತಿರುಗಿ -1; }
    DIR* dir;
    ರಚನೆ dirent* ನಮೂದು;
    ಪೂರ್ಣಾಂಕ max_ಹೆಜ್ಜೆ = -1;
    dir = opendir(output_log_dir);
    ಆದರೆ (dir == ಖಾಲಿ) { ಹಿಂತಿರುಗಿ -1; }
    ಆಗ ((ನಮೂದು = readdir(dir)) != ಖಾಲಿ) {
        ಆದರೆ (ಅಕ್ಷರಸರಣಿ_ಹೋಲಿಕೆ(ನಮೂದು->d_name, "DONE_", 5) == 0) {
            ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = atoi(ನಮೂದು->d_name + 5);
            ಆದರೆ (ಹೆಜ್ಜೆ > max_ಹೆಜ್ಜೆ) {
                max_ಹೆಜ್ಜೆ = ಹೆಜ್ಜೆ;
            }
        }
    }
    closedir(dir);
    ಹಿಂತಿರುಗಿ max_ಹೆಜ್ಜೆ;
}

ಬಾಹ್ಯ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ ತುದಿಗಳು_with_bin(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    ಆದರೆ (str == ಖಾಲಿ) { ಹಿಂತಿರುಗಿ 0; }
    size_t len = ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(str);
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* suffix = ".bin";
    size_t suffix_len = ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(suffix);
    ಆದರೆ (len < suffix_len) { ಹಿಂತಿರುಗಿ 0; }
    ಪೂರ್ಣಾಂಕ suffix_matches = ಅಕ್ಷರಸರಣಿ_ಹೋಲಿಕೆ(str + len - suffix_len, suffix, suffix_len) == 0;
    ಹಿಂತಿರುಗಿ suffix_matches;
}

#ಆದರೆ_ಅಂತ್ಯ