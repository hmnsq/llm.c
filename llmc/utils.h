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

बाह्य ओळीत खाता *fopen_check(स्थिरांक अक्षर *path, स्थिरांक अक्षर *mode, स्थिरांक अक्षर *file, पूर्णांक ओळ) {
    खाता *fp = ख_खोलो(path, mode);
    जर (fp == रिक्त) {
        ख_लिखो(मानक_त्रुटि, "Error: Failed to open file '%s' at %s:%d\n", path, file, ओळ);
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", ओळ);
        ख_लिखो(मानक_त्रुटि, "  Path: %s\n", path);
        ख_लिखो(मानक_त्रुटि, "  Mode: %s\n", mode);
        ख_लिखो(मानक_त्रुटि, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        ख_लिखो(मानक_त्रुटि, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        निकास(निकास_त्रुटि);
    }
    परत fp;
}

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

बाह्य ओळीत शून्य fread_check(शून्य *ptr, माप_प्रकार size, माप_प्रकार nmemb, खाता *stream, स्थिरांक अक्षर *file, पूर्णांक ओळ) {
    माप_प्रकार result = ख_पढ़ो(ptr, size, nmemb, stream);
    जर (result != nmemb) {
        जर (ख_पूर्ण(stream)) {
            ख_लिखो(मानक_त्रुटि, "Error: Unexpected end of file at %s:%d\n", file, ओळ);
        } अन्यथा जर (ख_त्रुटि(stream)) {
            ख_लिखो(मानक_त्रुटि, "Error: File read error at %s:%d\n", file, ओळ);
        } अन्यथा {
            ख_लिखो(मानक_त्रुटि, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, ओळ, nmemb, result);
        }
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", ओळ);
        ख_लिखो(मानक_त्रुटि, "  Expected elements: %zu\n", nmemb);
        ख_लिखो(मानक_त्रुटि, "  Read elements: %zu\n", result);
        निकास(निकास_त्रुटि);
    }
}

#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

बाह्य ओळीत शून्य fclose_check(खाता *fp, स्थिरांक अक्षर *file, पूर्णांक ओळ) {
    जर (ख_बंद(fp) != 0) {
        ख_लिखो(मानक_त्रुटि, "Error: Failed to close file at %s:%d\n", file, ओळ);
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", ओळ);
        निकास(निकास_त्रुटि);
    }
}

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

बाह्य ओळीत शून्य sclose_check(पूर्णांक sockfd, स्थिरांक अक्षर *file, पूर्णांक ओळ) {
    जर (बंद(sockfd) != 0) {
        ख_लिखो(मानक_त्रुटि, "Error: Failed to close socket at %s:%d\n", file, ओळ);
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", ओळ);
        निकास(निकास_त्रुटि);
    }
}

#define scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#ifdef _WIN32
बाह्य ओळीत शून्य closesocket_check(पूर्णांक sockfd, स्थिरांक अक्षर *file, पूर्णांक ओळ) {
    जर (closesocket(sockfd) != 0) {
        ख_लिखो(मानक_त्रुटि, "Error: Failed to close socket at %s:%d\n", file, ओळ);
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", ओळ);
        निकास(निकास_त्रुटि);
    }
}

#define closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#endif

बाह्य ओळीत शून्य fseek_check(खाता *fp, दीर्घ off, पूर्णांक whence, स्थिरांक अक्षर *file, पूर्णांक ओळ) {
    जर (ख_जाओ(fp, off, whence) != 0) {
        ख_लिखो(मानक_त्रुटि, "Error: Failed to seek in file at %s:%d\n", file, ओळ);
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  Offset: %ld\n", off);
        ख_लिखो(मानक_त्रुटि, "  Whence: %d\n", whence);
        ख_लिखो(मानक_त्रुटि, "  File:   %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line:   %d\n", ओळ);
        निकास(निकास_त्रुटि);
    }
}

#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

बाह्य ओळीत शून्य fwrite_check(शून्य *ptr, माप_प्रकार size, माप_प्रकार nmemb, खाता *stream, स्थिरांक अक्षर *file, पूर्णांक ओळ) {
    माप_प्रकार result = ख_डालो(ptr, size, nmemb, stream);
    जर (result != nmemb) {
        जर (ख_पूर्ण(stream)) {
            ख_लिखो(मानक_त्रुटि, "Error: Unexpected end of file at %s:%d\n", file, ओळ);
        } अन्यथा जर (ख_त्रुटि(stream)) {
            ख_लिखो(मानक_त्रुटि, "Error: File write error at %s:%d\n", file, ओळ);
        } अन्यथा {
            ख_लिखो(मानक_त्रुटि, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, ओळ, nmemb, result);
        }
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", ओळ);
        ख_लिखो(मानक_त्रुटि, "  Expected elements: %zu\n", nmemb);
        ख_लिखो(मानक_त्रुटि, "  Written elements: %zu\n", result);
        निकास(निकास_त्रुटि);
    }
}

#define fwriteCheck(ptr, size, nmemb, stream) fwrite_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

बाह्य ओळीत शून्य *malloc_check(माप_प्रकार size, स्थिरांक अक्षर *file, पूर्णांक ओळ) {
    शून्य *ptr = दो_स्मृति(size);
    जर (ptr == रिक्त) {
        ख_लिखो(मानक_त्रुटि, "Error: Memory allocation failed at %s:%d\n", file, ओळ);
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", ओळ);
        ख_लिखो(मानक_त्रुटि, "  Size: %zu bytes\n", size);
        निकास(निकास_त्रुटि);
    }
    परत ptr;
}

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
बाह्य ओळीत शून्य token_check(स्थिरांक पूर्णांक* tokens, पूर्णांक token_मोजणी, पूर्णांक vocab_size, स्थिरांक अक्षर *file, पूर्णांक ओळ) {
    साठी(पूर्णांक i = 0; i < token_मोजणी; i++) {
        जर(!(0 <= tokens[i] && tokens[i] < vocab_size)) {
            ख_लिखो(मानक_त्रुटि, "Error: Token out of vocabulary at %s:%d\n", file, ओळ);
            ख_लिखो(मानक_त्रुटि, "Error details:\n");
            ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
            ख_लिखो(मानक_त्रुटि, "  Line: %d\n", ओळ);
            ख_लिखो(मानक_त्रुटि, "  Token: %d\n", tokens[i]);
            ख_लिखो(मानक_त्रुटि, "  Position: %d\n", i);
            ख_लिखो(मानक_त्रुटि, "  Vocab: %d\n", vocab_size);
            निकास(निकास_त्रुटि);
        }
    }
}
#define tokenCheck(tokens, मोजणी, vocab) token_check(tokens, मोजणी, vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

बाह्य ओळीत शून्य create_dir_if_not_exists(स्थिरांक अक्षर *dir) {
    जर (dir == रिक्त) { परत; }
    संरचना stat st = {0};
    जर (stat(dir, &st) == -1) {
        जर (सूची_गढ़ो(dir, 0700) == -1) {
            म_लिखो("ERROR: could not create directory: %s\n", dir);
            निकास(निकास_त्रुटि);
        }
        म_लिखो("created directory: %s\n", dir);
    }
}

बाह्य ओळीत पूर्णांक find_max_step(स्थिरांक अक्षर* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    जर (output_log_dir == रिक्त) { परत -1; }
    सूची* dir;
    संरचना dirent* entry;
    पूर्णांक max_step = -1;
    dir = सूची_खोलो(output_log_dir);
    जर (dir == रिक्त) { परत -1; }
    पर्यंत ((entry = सूची_पढ़ो(dir)) != रिक्त) {
        जर (म_भेदन(entry->d_name, "DONE_", 5) == 0) {
            पूर्णांक step = म_से_प(entry->d_name + 5);
            जर (step > max_step) {
                max_step = step;
            }
        }
    }
    बंद_सूची(dir);
    परत max_step;
}

बाह्य ओळीत पूर्णांक ends_with_bin(स्थिरांक अक्षर* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    जर (str == रिक्त) { परत 0; }
    माप_प्रकार len = म_माप(str);
    स्थिरांक अक्षर* suffix = ".bin";
    माप_प्रकार suffix_len = म_माप(suffix);
    जर (len < suffix_len) { परत 0; }
    पूर्णांक suffix_matches = म_भेदन(str + len - suffix_len, suffix, suffix_len) == 0;
    परत suffix_matches;
}

#endif