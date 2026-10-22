/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#अगर_अघोषित UTILS_H
#घोषणा UTILS_H

#समावेश <unistd.h>
#समावेश <string.h>
#समावेश <stdio.h>
#समावेश <stdlib.h>
#समावेश <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#अगर_अघोषित _WIN32
#समावेश <dirent.h>
#समावेश <arpa/inet.h>
#पूर्ण_अगर

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

बाह्य अंतरभूत खाता *fopen_check(स्थिर अक्षर *path, स्थिर अक्षर *mode, स्थिर अक्षर *file, पूर्णांक पंक्ति) {
    खाता *fp = ख_खोलो(path, mode);
    अगर (fp == शून्य) {
        ख_लिखो(मानक_त्रुटि, "Error: Failed to open file '%s' at %s:%d\n", path, file, पंक्ति);
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", पंक्ति);
        ख_लिखो(मानक_त्रुटि, "  Path: %s\n", path);
        ख_लिखो(मानक_त्रुटि, "  Mode: %s\n", mode);
        ख_लिखो(मानक_त्रुटि, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        ख_लिखो(मानक_त्रुटि, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        निकास(निकास_त्रुटि);
    }
    वापस fp;
}

#घोषणा fopenCheck(path, mode) fopen_check(path, mode, __खाता__, __LINE__)

बाह्य अंतरभूत व्योम fread_check(व्योम *ptr, माप_प्रकार आकार, माप_प्रकार nmemb, खाता *stream, स्थिर अक्षर *file, पूर्णांक पंक्ति) {
    माप_प्रकार परिणाम = ख_पढ़ो(ptr, आकार, nmemb, stream);
    अगर (परिणाम != nmemb) {
        अगर (ख_पूर्ण(stream)) {
            ख_लिखो(मानक_त्रुटि, "Error: Unexpected end of file at %s:%d\n", file, पंक्ति);
        } अन्यथा अगर (ख_त्रुटि(stream)) {
            ख_लिखो(मानक_त्रुटि, "Error: File read error at %s:%d\n", file, पंक्ति);
        } अन्यथा {
            ख_लिखो(मानक_त्रुटि, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, पंक्ति, nmemb, परिणाम);
        }
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", पंक्ति);
        ख_लिखो(मानक_त्रुटि, "  Expected elements: %zu\n", nmemb);
        ख_लिखो(मानक_त्रुटि, "  Read elements: %zu\n", परिणाम);
        निकास(निकास_त्रुटि);
    }
}

#घोषणा freadCheck(ptr, आकार, nmemb, stream) fread_check(ptr, आकार, nmemb, stream, __खाता__, __LINE__)

बाह्य अंतरभूत व्योम fclose_check(खाता *fp, स्थिर अक्षर *file, पूर्णांक पंक्ति) {
    अगर (ख_बंद(fp) != 0) {
        ख_लिखो(मानक_त्रुटि, "Error: Failed to close file at %s:%d\n", file, पंक्ति);
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", पंक्ति);
        निकास(निकास_त्रुटि);
    }
}

#घोषणा fcloseCheck(fp) fclose_check(fp, __खाता__, __LINE__)

बाह्य अंतरभूत व्योम sclose_check(पूर्णांक sockfd, स्थिर अक्षर *file, पूर्णांक पंक्ति) {
    अगर (बंद(sockfd) != 0) {
        ख_लिखो(मानक_त्रुटि, "Error: Failed to close socket at %s:%d\n", file, पंक्ति);
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", पंक्ति);
        निकास(निकास_त्रुटि);
    }
}

#घोषणा scloseCheck(sockfd) sclose_check(sockfd, __खाता__, __LINE__)

#अगर_घोषित _WIN32
बाह्य अंतरभूत व्योम closesocket_check(पूर्णांक sockfd, स्थिर अक्षर *file, पूर्णांक पंक्ति) {
    अगर (closesocket(sockfd) != 0) {
        ख_लिखो(मानक_त्रुटि, "Error: Failed to close socket at %s:%d\n", file, पंक्ति);
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", पंक्ति);
        निकास(निकास_त्रुटि);
    }
}

#घोषणा closesocketCheck(sockfd) closesocket_check(sockfd, __खाता__, __LINE__)
#पूर्ण_अगर

बाह्य अंतरभूत व्योम fseek_check(खाता *fp, दीर्घ off, पूर्णांक whence, स्थिर अक्षर *file, पूर्णांक पंक्ति) {
    अगर (ख_जाओ(fp, off, whence) != 0) {
        ख_लिखो(मानक_त्रुटि, "Error: Failed to seek in file at %s:%d\n", file, पंक्ति);
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  Offset: %ld\n", off);
        ख_लिखो(मानक_त्रुटि, "  Whence: %d\n", whence);
        ख_लिखो(मानक_त्रुटि, "  File:   %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line:   %d\n", पंक्ति);
        निकास(निकास_त्रुटि);
    }
}

#घोषणा fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __खाता__, __LINE__)

बाह्य अंतरभूत व्योम fwrite_check(व्योम *ptr, माप_प्रकार आकार, माप_प्रकार nmemb, खाता *stream, स्थिर अक्षर *file, पूर्णांक पंक्ति) {
    माप_प्रकार परिणाम = ख_डालो(ptr, आकार, nmemb, stream);
    अगर (परिणाम != nmemb) {
        अगर (ख_पूर्ण(stream)) {
            ख_लिखो(मानक_त्रुटि, "Error: Unexpected end of file at %s:%d\n", file, पंक्ति);
        } अन्यथा अगर (ख_त्रुटि(stream)) {
            ख_लिखो(मानक_त्रुटि, "Error: File write error at %s:%d\n", file, पंक्ति);
        } अन्यथा {
            ख_लिखो(मानक_त्रुटि, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n",
                    file, पंक्ति, nmemb, परिणाम);
        }
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", पंक्ति);
        ख_लिखो(मानक_त्रुटि, "  Expected elements: %zu\n", nmemb);
        ख_लिखो(मानक_त्रुटि, "  Written elements: %zu\n", परिणाम);
        निकास(निकास_त्रुटि);
    }
}

#घोषणा fwriteCheck(ptr, आकार, nmemb, stream) fwrite_check(ptr, आकार, nmemb, stream, __खाता__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

बाह्य अंतरभूत व्योम *malloc_check(माप_प्रकार आकार, स्थिर अक्षर *file, पूर्णांक पंक्ति) {
    व्योम *ptr = दो_स्मृति(आकार);
    अगर (ptr == शून्य) {
        ख_लिखो(मानक_त्रुटि, "Error: Memory allocation failed at %s:%d\n", file, पंक्ति);
        ख_लिखो(मानक_त्रुटि, "Error details:\n");
        ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
        ख_लिखो(मानक_त्रुटि, "  Line: %d\n", पंक्ति);
        ख_लिखो(मानक_त्रुटि, "  Size: %zu bytes\n", आकार);
        निकास(निकास_त्रुटि);
    }
    वापस ptr;
}

#घोषणा mallocCheck(आकार) malloc_check(आकार, __खाता__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
बाह्य अंतरभूत व्योम token_check(स्थिर पूर्णांक* tokens, पूर्णांक token_गिनती, पूर्णांक vocab_आकार, स्थिर अक्षर *file, पूर्णांक पंक्ति) {
    क्रम(पूर्णांक i = 0; i < token_गिनती; i++) {
        अगर(!(0 <= tokens[i] && tokens[i] < vocab_आकार)) {
            ख_लिखो(मानक_त्रुटि, "Error: Token out of vocabulary at %s:%d\n", file, पंक्ति);
            ख_लिखो(मानक_त्रुटि, "Error details:\n");
            ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
            ख_लिखो(मानक_त्रुटि, "  Line: %d\n", पंक्ति);
            ख_लिखो(मानक_त्रुटि, "  Token: %d\n", tokens[i]);
            ख_लिखो(मानक_त्रुटि, "  Position: %d\n", i);
            ख_लिखो(मानक_त्रुटि, "  Vocab: %d\n", vocab_आकार);
            निकास(निकास_त्रुटि);
        }
    }
}
#घोषणा tokenCheck(tokens, गिनती, vocab) token_check(tokens, गिनती, vocab, __खाता__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

बाह्य अंतरभूत व्योम create_dir_if_not_exists(स्थिर अक्षर *dir) {
    अगर (dir == शून्य) { वापस; }
    काष्ठा stat st = {0};
    अगर (stat(dir, &st) == -1) {
        अगर (सूची_गढ़ो(dir, 0700) == -1) {
            म_लिखो("ERROR: could not create directory: %s\n", dir);
            निकास(निकास_त्रुटि);
        }
        म_लिखो("created directory: %s\n", dir);
    }
}

बाह्य अंतरभूत पूर्णांक find_max_step(स्थिर अक्षर* output_log_dir) {
    // find the DONE file in the log dir with highest step count
    अगर (output_log_dir == शून्य) { वापस -1; }
    सूची* dir;
    काष्ठा dirent* entry;
    पूर्णांक max_step = -1;
    dir = सूची_खोलो(output_log_dir);
    अगर (dir == शून्य) { वापस -1; }
    जबतक ((entry = सूची_पढ़ो(dir)) != शून्य) {
        अगर (म_भेदन(entry->d_name, "DONE_", 5) == 0) {
            पूर्णांक step = म_से_प(entry->d_name + 5);
            अगर (step > max_step) {
                max_step = step;
            }
        }
    }
    बंद_सूची(dir);
    वापस max_step;
}

बाह्य अंतरभूत पूर्णांक ends_with_bin(स्थिर अक्षर* str) {
    // checks if str ends with ".bin". could be generalized in the future.
    अगर (str == शून्य) { वापस 0; }
    माप_प्रकार len = म_माप(str);
    स्थिर अक्षर* suffix = ".bin";
    माप_प्रकार suffix_len = म_माप(suffix);
    अगर (len < suffix_len) { वापस 0; }
    पूर्णांक suffix_matches = म_भेदन(str + len - suffix_len, suffix, suffix_len) == 0;
    वापस suffix_matches;
}

#पूर्ण_अगर