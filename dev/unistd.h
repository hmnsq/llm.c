// header file that is necessary to compile on Windows
#ifndef UNISTD_H
#define UNISTD_H

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define WIN32_LEAN_AND_MEAN

#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <time.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h> // for malloc and free
#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <direct.h> // for _mkdir and _stat
#ಅನ್ತರ್ಭಾವ <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#pragma comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#ಅನ್ತರ್ಭಾವ <winsock2.h>

#define CLOCK_MONOTONIC 0
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ clock_gettime(ಪೂರ್ಣಾಙ್ಕ ignore_variable, ಸಂರಚನಾ timespec* tv)
{
    ಪ್ರತ್ಯಾಗಮ timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#define OMP /* turn it on */
#define F_OK 0
#define access _access

#define TURN_OFF_FP_FAST __pragma(float_control( precise, on, ತಳ್ಳು )) // Save current setting and turn on /fp:precise
#define TURN_ON_FP_FAST  __pragma(float_control(ತೆಗೆ)) // Restore file's default settings

#define mkdir(path, mode) _mkdir(path) /* sketchy way to get mkdir to work on windows */
#define stat _stat

ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ glob_t {
    size_t gl_pathc;    // Count of matched pathnames
    ವರ್ಣ **gl_pathv;    // List of matched pathnames
} glob_t;

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ replace_forward_slashes(ವರ್ಣ* str) {
    ಯಾವತ್ (*str) {
        ಯದಿ (*str == '/') {
            *str = '\\';
        }
        str++;
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ globfree(glob_t *pglob) {
    ಪ್ರತಿ (size_t i = 0; i < pglob->gl_pathc; ++i) {
        ಮುಕ್ತ(pglob->gl_pathv[i]); // Free the allocated memory for each filename
    }
    ಮುಕ್ತ(pglob->gl_pathv); // Free the allocated memory for the list of filenames
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ glob(ಧ್ರುವ ವರ್ಣ* pattern, ಪೂರ್ಣಾಙ್ಕ ignored_flags, ಪೂರ್ಣಾಙ್ಕ (*ignored_errfunc)(ಧ್ರುವ ವರ್ಣ* epath, ಪೂರ್ಣಾಙ್ಕ eerrno), glob_t* pglob){
    ಸಂರಚನಾ _finddata_t find_file_data;
    ವರ್ಣ full_path[576]; // stored in pglob->gl_pathv[n]
    ವರ್ಣ directory_path[512] = {0}; // Store the directory path from the pattern
    ವರ್ಣ pattern_copy[512]; // Copy of the pattern to modify

    strncpy_s(pattern_copy, ಪರಿಮಾಣ(pattern_copy) - 1, pattern, ಪರಿಮಾಣ(pattern_copy) - 1);

    replace_forward_slashes (pattern_copy); // Replace forward slashes with backslashes

    ಯದಿ (strchr(pattern_copy, '\\') != (ಶೂನ್ಯ*) ರಿಕ್ತ) {
        strncpy_s(directory_path, ಪರಿಮಾಣ(directory_path) - 1, pattern_copy, strrchr(pattern_copy, '\\') - pattern_copy + 1);
        directory_path[strrchr(pattern_copy, '\\') - pattern_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t find_handle = _findfirst(pattern_copy, &find_file_data);

    ಯದಿ (find_handle == -1) {
        ಪ್ರತ್ಯಾಗಮ 1; // No files found
    }

    size_t file_ಎಣಿಕೆ = 0;
    size_t max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (ವರ್ಣ **) malloc(max_files * ಪರಿಮಾಣ(ವರ್ಣ*)); // freed in globfree

    ಯದಿ (pglob->gl_pathv == ರಿಕ್ತ) {
        _findclose(find_handle);
        ಪ್ರತ್ಯಾಗಮ 2; // Memory allocation failed
    }

    ಕುರು {
        ಯದಿ (file_ಎಣಿಕೆ >= max_files) {
            _findclose(find_handle);
            ಪ್ರತ್ಯಾಗಮ 2; // Too many files found
            }

        snprintf(full_path, ಪರಿಮಾಣ(full_path), "%s%s", directory_path, find_file_data.name);

        pglob->gl_pathv[file_ಎಣಿಕೆ] = _strdup(full_path); // freed in globfree

        ಯದಿ (pglob->gl_pathv[file_ಎಣಿಕೆ] == ರಿಕ್ತ) {
            _findclose(find_handle);
            ಪ್ರತ್ಯಾಗಮ 2; // Memory allocation for filename failed
        }
        file_ಎಣಿಕೆ++;
    } ಯಾವತ್ (_findnext(find_handle, &find_file_data) == 0);

    _findclose(find_handle);

    pglob->gl_pathc = file_ಎಣಿಕೆ;
    ಪ್ರತ್ಯಾಗಮ 0;
}

// dirent.h support

#define MAX_PATH_LENGTH 512
ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ dirent {
    ವರ್ಣ d_name[MAX_PATH_LENGTH];
} dirent;

ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ DIR {
    intptr_t handle;
    ಸಂರಚನಾ _finddata_t findFileData;
    ಪೂರ್ಣಾಙ್ಕ firstRead;
} DIR;

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ DIR *opendir(ಧ್ರುವ ವರ್ಣ *name) {
    DIR *dir = (DIR *)malloc(ಪರಿಮಾಣ(DIR));
    ಯದಿ (dir == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ವರ್ಣ ಹುಡುಕು_Path[MAX_PATH_LENGTH];

    snprintf(ಹುಡುಕು_Path, MAX_PATH_LENGTH, "%s\\*.*", name);

    dir->handle = _findfirst(ಹುಡುಕು_Path, &dir->findFileData);
    ಯದಿ (dir->handle == -1) {
        ಮುಕ್ತ(dir);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    dir->firstRead = 1;
    ಪ್ರತ್ಯಾಗಮ dir;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಸಂರಚನಾ dirent *readdir(DIR *directory) {
    ಸ್ಥಿರ ಸಂರಚನಾ dirent result;

    ಯದಿ (directory->firstRead) {
        directory->firstRead = 0;
    } ಅನ್ಯಥಾ {
        ಯದಿ (_findnext(directory->handle, &directory->findFileData) != 0) {
            ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
        }
    }

    strncpy(result.d_name, directory->findFileData.name, MAX_PATH_LENGTH);
    result.d_name[MAX_PATH_LENGTH - 1] = '\0'; // Ensure null termination
    ಪ್ರತ್ಯಾಗಮ &result;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ closedir(DIR *directory) {
    ಯದಿ (directory == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಯದಿ (_findclose(directory->handle) != 0) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಮುಕ್ತ(directory);
    ಪ್ರತ್ಯಾಗಮ 0;
}
#endif // UNISTD_H
