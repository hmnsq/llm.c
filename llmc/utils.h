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

बाह्य अन्तःस्थ खाता *fopen_check(अचर वर्ण *path, अचर वर्ण *mode, अचर वर्ण *file, पूर्णाङ्क पङ्क्ति) {
    खाता *fp = ख_खोलो(path, mode);
    यदि (fp == रिक्त) {
        ख_लिखो(मानक_त्रुटि, "Error: Failed to open file '%s' at %s:%d\n", path, file, पङ्क्ति);
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", पङ्क्ति);
        ख_लिखो(मानक_त्रुटि, "  Path: %s\n", path);
        ख_लिखो(मानक_त्रुटि, "  Mode: %s\n", mode);
        ख_लिखो(मानक_त्रुटि, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        ख_लिखो(मानक_त्रुटि, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        निकास(निकास_त्रुटि);
    }
    फर्काऊ fp;
}

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

बाह्य अन्तःस्थ शून्य fread_check(शून्य *ptr, माप_प्रकार size, माप_प्रकार nmemb, खाता *stream, अचर वर्ण *file, पूर्णाङ्क पङ्क्ति) {
    माप_प्रकार result = ख_पढ़ो(ptr, size, nmemb, stream);
    यदि (result != nmemb) {
        यदि (ख_पूर्ण(stream)) {
            ख_लिखो(मानक_त्रुटि, "Error: Unexpected end of file at %s:%d\n", file, पङ्क्ति);
        } अन्यथा यदि (ख_त्रुटि(stream)) {
            ख_लिखो(मानक_त्रुटि, "Error: File read error at %s:%d\n", file, पङ्क्ति);
        } अन्यथा {
            ख_लिखो(मानक_त्रुटि, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, पङ्क्ति, nmemb, result);
        }
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", पङ्क्ति);
        ख_लिखो(मानक_त्रुटि, "  Expected elements: %zu\n", nmemb);
        ख_लिखो(मानक_त्रुटि, "  Read elements: %zu\n", result);
        निकास(निकास_त्रुटि);
    }
}

#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

बाह्य अन्तःस्थ शून्य fclose_check(खाता *fp, अचर वर्ण *file, पूर्णाङ्क पङ्क्ति) {
    यदि (ख_बंद(fp) != 0) {
        ख_लिखो(मानक_त्रुटि, "Error: Failed to close file at %s:%d\n", file, पङ्क्ति);
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", पङ्क्ति);
        निकास(निकास_त्रुटि);
    }
}

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

बाह्य अन्तःस्थ शून्य sclose_check(पूर्णाङ्क sockfd, अचर वर्ण *file, पूर्णाङ्क पङ्क्ति) {
    यदि (बंद(sockfd) != 0) {
        ख_लिखो(मानक_त्रुटि, "Error: Failed to close socket at %s:%d\n", file, पङ्क्ति);
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", पङ्क्ति);
        निकास(निकास_त्रुटि);
    }
}

#define scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#ifdef _WIN32
बाह्य अन्तःस्थ शून्य closesocket_check(पूर्णाङ्क sockfd, अचर वर्ण *file, पूर्णाङ्क पङ्क्ति) {
    यदि (closesocket(sockfd) != 0) {
        ख_लिखो(मानक_त्रुटि, "Error: Failed to close socket at %s:%d\n", file, पङ्क्ति);
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", पङ्क्ति);
        निकास(निकास_त्रुटि);
    }
}

#define closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#endif

बाह्य अन्तःस्थ शून्य fseek_check(खाता *fp, दीर्घ off, पूर्णाङ्क whence, अचर वर्ण *file, पूर्णाङ्क पङ्क्ति) {
    यदि (ख_जाओ(fp, off, whence) != 0) {
        ख_लिखो(मानक_त्रुटि, "Error: Failed to seek in file at %s:%d\n", file, पङ्क्ति);
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  Offset: %ld\n", off);
        ख_लिखो(मानक_त्रुटि, "  Whence: %d\n", whence);
        ख_लिखो(मानक_त्रुटि, "  File:   %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line:   %d\n", पङ्क्ति);
        निकास(निकास_त्रुटि);
    }
}

#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

बाह्य अन्तःस्थ शून्य fwrite_check(शून्य *ptr, माप_प्रकार size, माप_प्रकार nmemb, खाता *stream, अचर वर्ण *file, पूर्णाङ्क पङ्क्ति) {
    माप_प्रकार result = ख_डालो(ptr, size, nmemb, stream);
    यदि (result != nmemb) {
        यदि (ख_पूर्ण(stream)) {
            ख_लिखो(मानक_त्रुटि, "Error: Unexpected end of file at %s:%d\n", file, पङ्क्ति);
        } अन्यथा यदि (ख_त्रुटि(stream)) {
            ख_लिखो(मानक_त्रुटि, "Error: File write error at %s:%d\n", file, पङ्क्ति);
        } अन्यथा {
            ख_लिखो(मानक_त्रुटि, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, पङ्क्ति, nmemb, result);
        }
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", पङ्क्ति);
        ख_लिखो(मानक_त्रुटि, "  Expected elements: %zu\n", nmemb);
        ख_लिखो(मानक_त्रुटि, "  Written elements: %zu\n", result);
        निकास(निकास_त्रुटि);
    }
}

#define fwriteCheck(ptr, size, nmemb, stream) fwrite_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

बाह्य अन्तःस्थ शून्य *malloc_check(माप_प्रकार size, अचर वर्ण *file, पूर्णाङ्क पङ्क्ति) {
    शून्य *ptr = दो_स्मृति(size);
    यदि (ptr == रिक्त) {
        ख_लिखो(मानक_त्रुटि, "Error: Memory allocation failed at %s:%d\n", file, पङ्क्ति);
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", पङ्क्ति);
        ख_लिखो(मानक_त्रुटि, "  Size: %zu bytes\n", size);
        निकास(निकास_त्रुटि);
    }
    फर्काऊ ptr;
}

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
बाह्य अन्तःस्थ शून्य token_check(अचर पूर्णाङ्क* tokens, पूर्णाङ्क token_गन्ती, पूर्णाङ्क vocab_size, अचर वर्ण *file, पूर्णाङ्क पङ्क्ति) {
    लागि(पूर्णाङ्क i = 0; i < token_गन्ती; i++) {
        यदि(!(0 <= tokens[i] && tokens[i] < vocab_size)) {
            ख_लिखो(मानक_त्रुटि, "Error: Token out of vocabulary at %s:%d\n", file, पङ्क्ति);
            ख_लिखो(मानक_त्रुटि, "Error details:\n");
            ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
            ख_लिखो(मानक_त्रुटि, "  Line: %d\n", पङ्क्ति);
            ख_लिखो(मानक_त्रुटि, "  Token: %d\n", tokens[i]);
            ख_लिखो(मानक_त्रुटि, "  Position: %d\n", i);
            ख_लिखो(मानक_त्रुटि, "  Vocab: %d\n", vocab_size);
            निकास(निकास_त्रुटि);
        }
    }
}
#define tokenCheck(tokens, गन्ती, vocab) token_check(tokens, गन्ती, vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

बाह्य अन्तःस्थ शून्य create_dir_if_not_exists(अचर वर्ण *dir) {
    यदि (dir == रिक्त) { फर्काऊ; }
    संरचना stat st = {0};
    यदि (stat(dir, &st) == -1) {
        यदि (सूची_गढ़ो(dir, 0700) == -1) {
            म_लिखो("ERROR: could not create directory: %s\n", dir);
            निकास(निकास_त्रुटि);
        }
        म_लिखो("created directory: %s\n", dir);
    }
}

बाह्य अन्तःस्थ पूर्णाङ्क find_max_step(अचर वर्ण* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    यदि (output_log_dir == रिक्त) { फर्काऊ -1; }
    सूची* dir;
    संरचना dirent* entry;
    पूर्णाङ्क max_step = -1;
    dir = सूची_खोलो(output_log_dir);
    यदि (dir == रिक्त) { फर्काऊ -1; }
    जबसम्म ((entry = सूची_पढ़ो(dir)) != रिक्त) {
        यदि (म_भेदन(entry->d_name, "DONE_", 5) == 0) {
            पूर्णाङ्क step = म_से_प(entry->d_name + 5);
            यदि (step > max_step) {
                max_step = step;
            }
        }
    }
    बंद_सूची(dir);
    फर्काऊ max_step;
}

बाह्य अन्तःस्थ पूर्णाङ्क ends_with_bin(अचर वर्ण* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    यदि (str == रिक्त) { फर्काऊ 0; }
    माप_प्रकार len = म_माप(str);
    अचर वर्ण* suffix = ".bin";
    माप_प्रकार suffix_len = म_माप(suffix);
    यदि (len < suffix_len) { फर्काऊ 0; }
    पूर्णाङ्क suffix_matches = म_भेदन(str + len - suffix_len, suffix, suffix_len) == 0;
    फर्काऊ suffix_matches;
}

#endif