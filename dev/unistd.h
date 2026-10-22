// header file that is necessary to compile on Windows
#ifndef UNISTD_H
#define UNISTD_H

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define WIN32_LEAN_AND_MEAN

#अन्तर्भाव <stdio.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव <time.h>
#अन्तर्भाव <stdlib.h> // for malloc and free
#अन्तर्भाव <string.h>
#अन्तर्भाव <direct.h> // for _mkdir and _stat
#अन्तर्भाव <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#pragma comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#अन्तर्भाव <winsock2.h>

#define CLOCK_MONOTONIC 0
स्थिर अन्तःस्थ पूर्णाङ्क clock_gettime(पूर्णाङ्क ignore_variable, संरचना timespec* tv)
{
    प्रत्यागम timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#define OMP /* turn it on */
#define F_OK 0
#define access _access

#define TURN_OFF_FP_FAST __pragma(float_control( precise, on, ढकला )) // Save current setting and turn on /fp:precise
#define TURN_ON_FP_FAST  __pragma(float_control(काढा)) // Restore file's default settings

#define mkdir(path, mode) _mkdir(path) /* sketchy way to get mkdir to work on windows */
#define stat _stat

प्रकारनिर्देश संरचना glob_t {
    size_t gl_pathc;    // Count of matched pathnames
    वर्ण **gl_pathv;    // List of matched pathnames
} glob_t;

स्थिर अन्तःस्थ शून्य replace_forward_slashes(वर्ण* str) {
    यावत् (*str) {
        यदि (*str == '/') {
            *str = '\\';
        }
        str++;
    }
}

स्थिर अन्तःस्थ शून्य globfree(glob_t *pglob) {
    प्रति (size_t i = 0; i < pglob->gl_pathc; ++i) {
        मुक्त(pglob->gl_pathv[i]); // Free the allocated memory for each filename
    }
    मुक्त(pglob->gl_pathv); // Free the allocated memory for the list of filenames
}

स्थिर अन्तःस्थ पूर्णाङ्क glob(ध्रुव वर्ण* pattern, पूर्णाङ्क ignored_flags, पूर्णाङ्क (*ignored_errfunc)(ध्रुव वर्ण* epath, पूर्णाङ्क eerrno), glob_t* pglob){
    संरचना _finddata_t find_file_data;
    वर्ण full_path[576]; // stored in pglob->gl_pathv[n]
    वर्ण directory_path[512] = {0}; // Store the directory path from the pattern
    वर्ण pattern_copy[512]; // Copy of the pattern to modify

    strncpy_s(pattern_copy, परिमाण(pattern_copy) - 1, pattern, परिमाण(pattern_copy) - 1);

    replace_forward_slashes (pattern_copy); // Replace forward slashes with backslashes

    यदि (strchr(pattern_copy, '\\') != (शून्य*) रिक्त) {
        strncpy_s(directory_path, परिमाण(directory_path) - 1, pattern_copy, strrchr(pattern_copy, '\\') - pattern_copy + 1);
        directory_path[strrchr(pattern_copy, '\\') - pattern_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t find_handle = _findfirst(pattern_copy, &find_file_data);

    यदि (find_handle == -1) {
        प्रत्यागम 1; // No files found
    }

    size_t file_मोजणी = 0;
    size_t max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (वर्ण **) malloc(max_files * परिमाण(वर्ण*)); // freed in globfree

    यदि (pglob->gl_pathv == रिक्त) {
        _findclose(find_handle);
        प्रत्यागम 2; // Memory allocation failed
    }

    कुरु {
        यदि (file_मोजणी >= max_files) {
            _findclose(find_handle);
            प्रत्यागम 2; // Too many files found
            }

        snprintf(full_path, परिमाण(full_path), "%s%s", directory_path, find_file_data.name);

        pglob->gl_pathv[file_मोजणी] = _strdup(full_path); // freed in globfree

        यदि (pglob->gl_pathv[file_मोजणी] == रिक्त) {
            _findclose(find_handle);
            प्रत्यागम 2; // Memory allocation for filename failed
        }
        file_मोजणी++;
    } यावत् (_findnext(find_handle, &find_file_data) == 0);

    _findclose(find_handle);

    pglob->gl_pathc = file_मोजणी;
    प्रत्यागम 0;
}

// dirent.h support

#define MAX_PATH_LENGTH 512
प्रकारनिर्देश संरचना dirent {
    वर्ण d_name[MAX_PATH_LENGTH];
} dirent;

प्रकारनिर्देश संरचना DIR {
    intptr_t handle;
    संरचना _finddata_t findFileData;
    पूर्णाङ्क firstRead;
} DIR;

स्थिर अन्तःस्थ DIR *opendir(ध्रुव वर्ण *name) {
    DIR *dir = (DIR *)malloc(परिमाण(DIR));
    यदि (dir == रिक्त) {
        प्रत्यागम रिक्त;
    }

    वर्ण शोध_Path[MAX_PATH_LENGTH];

    snprintf(शोध_Path, MAX_PATH_LENGTH, "%s\\*.*", name);

    dir->handle = _findfirst(शोध_Path, &dir->findFileData);
    यदि (dir->handle == -1) {
        मुक्त(dir);
        प्रत्यागम रिक्त;
    }

    dir->firstRead = 1;
    प्रत्यागम dir;
}

स्थिर अन्तःस्थ संरचना dirent *readdir(DIR *directory) {
    स्थिर संरचना dirent result;

    यदि (directory->firstRead) {
        directory->firstRead = 0;
    } अन्यथा {
        यदि (_findnext(directory->handle, &directory->findFileData) != 0) {
            प्रत्यागम रिक्त;
        }
    }

    strncpy(result.d_name, directory->findFileData.name, MAX_PATH_LENGTH);
    result.d_name[MAX_PATH_LENGTH - 1] = '\0'; // Ensure null termination
    प्रत्यागम &result;
}

स्थिर अन्तःस्थ पूर्णाङ्क closedir(DIR *directory) {
    यदि (directory == रिक्त) {
        प्रत्यागम -1;
    }

    यदि (_findclose(directory->handle) != 0) {
        प्रत्यागम -1;
    }

    मुक्त(directory);
    प्रत्यागम 0;
}
#endif // UNISTD_H
