/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#నిర్వచితం_కాకపోతే UTILS_H
#నిర్వచించు UTILS_H

#అన్తర్భావ <unistd.h>
#అన్తర్భావ <string.h>
#అన్తర్భావ <stdio.h>
#అన్తర్భావ <stdlib.h>
#అన్తర్భావ <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#నిర్వచితం_కాకపోతే _WIN32
#అన్తర్భావ <dirent.h>
#అన్తర్భావ <arpa/inet.h>
#అయితే_ముగింపు

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

బాహ్య అంతఃవరుస FILE *fopen_check(స్థిరరాశి అక్షరం *మార్గం, స్థిరరాశి అక్షరం *బహుళకం, స్థిరరాశి అక్షరం *file, పూర్ణాంకం పంక్తి) {
    FILE *fp = fopen(మార్గం, బహుళకం);
    ఒకవేళ (fp == ఖాళీ) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", మార్గం, file, పంక్తి);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", పంక్తి);
        fprintf(stderr, "  Path: %s\n", మార్గం);
        fprintf(stderr, "  Mode: %s\n", బహుళకం);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        నిష్క్రమించు(EXIT_FAILURE);
    }
    తిరిగి fp;
}

#నిర్వచించు fopenCheck(మార్గం, బహుళకం) fopen_check(మార్గం, బహుళకం, __FILE__, __LINE__)

బాహ్య అంతఃవరుస శూన్య fread_check(శూన్య *ptr, size_t కొలత, size_t nmemb, FILE *ప్రవాహం, స్థిరరాశి అక్షరం *file, పూర్ణాంకం పంక్తి) {
    size_t ఫలితం = fread(ptr, కొలత, nmemb, ప్రవాహం);
    ఒకవేళ (ఫలితం != nmemb) {
        ఒకవేళ (feof(ప్రవాహం)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, పంక్తి);
        } లేకపోతే ఒకవేళ (ferror(ప్రవాహం)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, పంక్తి);
        } లేకపోతే {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, పంక్తి, nmemb, ఫలితం);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", పంక్తి);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", ఫలితం);
        నిష్క్రమించు(EXIT_FAILURE);
    }
}

#నిర్వచించు freadCheck(ptr, కొలత, nmemb, ప్రవాహం) fread_check(ptr, కొలత, nmemb, ప్రవాహం, __FILE__, __LINE__)

బాహ్య అంతఃవరుస శూన్య fclose_check(FILE *fp, స్థిరరాశి అక్షరం *file, పూర్ణాంకం పంక్తి) {
    ఒకవేళ (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, పంక్తి);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", పంక్తి);
        నిష్క్రమించు(EXIT_FAILURE);
    }
}

#నిర్వచించు fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

బాహ్య అంతఃవరుస శూన్య sclose_check(పూర్ణాంకం sockfd, స్థిరరాశి అక్షరం *file, పూర్ణాంకం పంక్తి) {
    ఒకవేళ (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, పంక్తి);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", పంక్తి);
        నిష్క్రమించు(EXIT_FAILURE);
    }
}

#నిర్వచించు scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#నిర్వచితమైతే _WIN32
బాహ్య అంతఃవరుస శూన్య closesocket_check(పూర్ణాంకం sockfd, స్థిరరాశి అక్షరం *file, పూర్ణాంకం పంక్తి) {
    ఒకవేళ (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, పంక్తి);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", పంక్తి);
        నిష్క్రమించు(EXIT_FAILURE);
    }
}

#నిర్వచించు closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#అయితే_ముగింపు

బాహ్య అంతఃవరుస శూన్య fseek_check(FILE *fp, దీర్ఘ off, పూర్ణాంకం whence, స్థిరరాశి అక్షరం *file, పూర్ణాంకం పంక్తి) {
    ఒకవేళ (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, పంక్తి);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", పంక్తి);
        నిష్క్రమించు(EXIT_FAILURE);
    }
}

#నిర్వచించు fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

బాహ్య అంతఃవరుస శూన్య fwrite_check(శూన్య *ptr, size_t కొలత, size_t nmemb, FILE *ప్రవాహం, స్థిరరాశి అక్షరం *file, పూర్ణాంకం పంక్తి) {
    size_t ఫలితం = fwrite(ptr, కొలత, nmemb, ప్రవాహం);
    ఒకవేళ (ఫలితం != nmemb) {
        ఒకవేళ (feof(ప్రవాహం)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, పంక్తి);
        } లేకపోతే ఒకవేళ (ferror(ప్రవాహం)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, పంక్తి);
        } లేకపోతే {
            fprintf(stderr, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, పంక్తి, nmemb, ఫలితం);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", పంక్తి);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Written elements: %zu\n", ఫలితం);
        నిష్క్రమించు(EXIT_FAILURE);
    }
}

#నిర్వచించు fwriteCheck(ptr, కొలత, nmemb, ప్రవాహం) fwrite_check(ptr, కొలత, nmemb, ప్రవాహం, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

బాహ్య అంతఃవరుస శూన్య *malloc_check(size_t కొలత, స్థిరరాశి అక్షరం *file, పూర్ణాంకం పంక్తి) {
    శూన్య *ptr = జ్ఞాపకం_ఇవ్వు(కొలత);
    ఒకవేళ (ptr == ఖాళీ) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, పంక్తి);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", పంక్తి);
        fprintf(stderr, "  Size: %zu bytes\n", కొలత);
        నిష్క్రమించు(EXIT_FAILURE);
    }
    తిరిగి ptr;
}

#నిర్వచించు mallocCheck(కొలత) malloc_check(కొలత, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
బాహ్య అంతఃవరుస శూన్య token_check(స్థిరరాశి పూర్ణాంకం* టోకెన్లు, పూర్ణాంకం token_లెక్క, పూర్ణాంకం vocab_కొలత, స్థిరరాశి అక్షరం *file, పూర్ణాంకం పంక్తి) {
    కొరకు(పూర్ణాంకం i = 0; i < token_లెక్క; i++) {
        ఒకవేళ(!(0 <= టోకెన్లు[i] && టోకెన్లు[i] < vocab_కొలత)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, పంక్తి);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", పంక్తి);
            fprintf(stderr, "  Token: %d\n", టోకెన్లు[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_కొలత);
            నిష్క్రమించు(EXIT_FAILURE);
        }
    }
}
#నిర్వచించు tokenCheck(టోకెన్లు, లెక్క, vocab) token_check(టోకెన్లు, లెక్క, vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

బాహ్య అంతఃవరుస శూన్య create_dir_if_not_exists(స్థిరరాశి అక్షరం *dir) {
    ఒకవేళ (dir == ఖాళీ) { తిరిగి; }
    నిర్మాణం stat st = {0};
    ఒకవేళ (stat(dir, &st) == -1) {
        ఒకవేళ (mkdir(dir, 0700) == -1) {
            రాయి("ERROR: could not create directory: %s\n", dir);
            నిష్క్రమించు(EXIT_FAILURE);
        }
        రాయి("created directory: %s\n", dir);
    }
}

బాహ్య అంతఃవరుస పూర్ణాంకం కనుగొను_max_అడుగు(స్థిరరాశి అక్షరం* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    ఒకవేళ (output_log_dir == ఖాళీ) { తిరిగి -1; }
    DIR* dir;
    నిర్మాణం dirent* నమోదు_2;
    పూర్ణాంకం max_అడుగు = -1;
    dir = opendir(output_log_dir);
    ఒకవేళ (dir == ఖాళీ) { తిరిగి -1; }
    అయితే ((నమోదు_2 = readdir(dir)) != ఖాళీ) {
        ఒకవేళ (అక్షరశ్రేణి_పోలిక(నమోదు_2->d_name, "DONE_", 5) == 0) {
            పూర్ణాంకం అడుగు = atoi(నమోదు_2->d_name + 5);
            ఒకవేళ (అడుగు > max_అడుగు) {
                max_అడుగు = అడుగు;
            }
        }
    }
    closedir(dir);
    తిరిగి max_అడుగు;
}

బాహ్య అంతఃవరుస పూర్ణాంకం చివర్లు_with_bin(స్థిరరాశి అక్షరం* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    ఒకవేళ (str == ఖాళీ) { తిరిగి 0; }
    size_t len = అక్షరశ్రేణి_పొడవు(str);
    స్థిరరాశి అక్షరం* suffix = ".bin";
    size_t suffix_len = అక్షరశ్రేణి_పొడవు(suffix);
    ఒకవేళ (len < suffix_len) { తిరిగి 0; }
    పూర్ణాంకం suffix_matches = అక్షరశ్రేణి_పోలిక(str + len - suffix_len, suffix, suffix_len) == 0;
    తిరిగి suffix_matches;
}

#అయితే_ముగింపు