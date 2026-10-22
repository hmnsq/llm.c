// header file that is necessary to compile on Windows
#ifndef UNISTD_H
#define UNISTD_H

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define WIN32_LEAN_AND_MEAN

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h> // for malloc and free
#include <string.h>
#include <direct.h> // for _mkdir and _stat
#include <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#pragma comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#include <winsock2.h>

#define CLOCK_MONOTONIC 0
static anta_hstha පූර්ණ_සංඛ්‍යා clock_gettime(පූර්ණ_සංඛ්‍යා ignore_variable, sa_nracanaa timespec* tv)
{
    ආපසු timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#define OMP /* turn it on */
#define F_OK 0
#define access _access

#define TURN_OFF_FP_FAST __pragma(float_control( precise, on, push )) // Save current setting and turn on /fp:precise
#define TURN_ON_FP_FAST  __pragma(float_control(pop)) // Restore file's default settings

#define mkdir(path, mode) _mkdir(path) /* sketchy way to get mkdir to work on windows */
#define stat _stat

typedef sa_nracanaa glob_t {
    size_t gl_pathc;    // Count of matched pathnames
    var_na **gl_pathv;    // List of matched pathnames
} glob_t;

static anta_hstha void replace_forward_slashes(var_na* str) {
    අතරතුර (*str) {
        නම් (*str == '/') {
            *str = '\\';
        }
        str++;
    }
}

static anta_hstha void globfree(glob_t *pglob) {
    සඳහා (size_t i = 0; i < pglob->gl_pathc; ++i) {
        free(pglob->gl_pathv[i]); // Free the allocated memory for each filename
    }
    free(pglob->gl_pathv); // Free the allocated memory for the list of filenames
}

static anta_hstha පූර්ණ_සංඛ්‍යා glob(const var_na* pattern, පූර්ණ_සංඛ්‍යා ignored_flags, පූර්ණ_සංඛ්‍යා (*ignored_errfunc)(const var_na* epath, පූර්ණ_සංඛ්‍යා eerrno), glob_t* pglob){
    sa_nracanaa _finddata_t find_file_data;
    var_na full_path[576]; // stored in pglob->gl_pathv[n]
    var_na directory_path[512] = {0}; // Store the directory path from the pattern
    var_na pattern_copy[512]; // Copy of the pattern to modify

    strncpy_s(pattern_copy, parimaa_na(pattern_copy) - 1, pattern, parimaa_na(pattern_copy) - 1);

    replace_forward_slashes (pattern_copy); // Replace forward slashes with backslashes

    නම් (strchr(pattern_copy, '\\') != (void*) NULL) {
        strncpy_s(directory_path, parimaa_na(directory_path) - 1, pattern_copy, strrchr(pattern_copy, '\\') - pattern_copy + 1);
        directory_path[strrchr(pattern_copy, '\\') - pattern_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t find_handle = _findfirst(pattern_copy, &find_file_data);

    නම් (find_handle == -1) {
        ආපසු 1; // No files found
    }

    size_t file_count = 0;
    size_t max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (var_na **) malloc(max_files * parimaa_na(var_na*)); // freed in globfree

    නම් (pglob->gl_pathv == NULL) {
        _findclose(find_handle);
        ආපසු 2; // Memory allocation failed
    }

    do {
        නම් (file_count >= max_files) {
            _findclose(find_handle);
            ආපසු 2; // Too many files found
            }

        snprintf(full_path, parimaa_na(full_path), "%s%s", directory_path, find_file_data.name);

        pglob->gl_pathv[file_count] = _strdup(full_path); // freed in globfree

        නම් (pglob->gl_pathv[file_count] == NULL) {
            _findclose(find_handle);
            ආපසු 2; // Memory allocation for filename failed
        }
        file_count++;
    } අතරතුර (_findnext(find_handle, &find_file_data) == 0);

    _findclose(find_handle);

    pglob->gl_pathc = file_count;
    ආපසු 0;
}

// dirent.h support

#define MAX_PATH_LENGTH 512
typedef sa_nracanaa dirent {
    var_na d_name[MAX_PATH_LENGTH];
} dirent;

typedef sa_nracanaa DIR {
    intptr_t handle;
    sa_nracanaa _finddata_t findFileData;
    පූර්ණ_සංඛ්‍යා firstRead;
} DIR;

static anta_hstha DIR *opendir(const var_na *name) {
    DIR *dir = (DIR *)malloc(parimaa_na(DIR));
    නම් (dir == NULL) {
        ආපසු NULL;
    }

    var_na searchPath[MAX_PATH_LENGTH];

    snprintf(searchPath, MAX_PATH_LENGTH, "%s\\*.*", name);

    dir->handle = _findfirst(searchPath, &dir->findFileData);
    නම් (dir->handle == -1) {
        free(dir);
        ආපසු NULL;
    }

    dir->firstRead = 1;
    ආපසු dir;
}

static anta_hstha sa_nracanaa dirent *readdir(DIR *directory) {
    static sa_nracanaa dirent result;

    නම් (directory->firstRead) {
        directory->firstRead = 0;
    } නැතහොත් {
        නම් (_findnext(directory->handle, &directory->findFileData) != 0) {
            ආපසු NULL;
        }
    }

    strncpy(result.d_name, directory->findFileData.name, MAX_PATH_LENGTH);
    result.d_name[MAX_PATH_LENGTH - 1] = '\0'; // Ensure null termination
    ආපසු &result;
}

static anta_hstha පූර්ණ_සංඛ්‍යා closedir(DIR *directory) {
    නම් (directory == NULL) {
        ආපසු -1;
    }

    නම් (_findclose(directory->handle) != 0) {
        ආපසු -1;
    }

    free(directory);
    ආපසු 0;
}
#endif // UNISTD_H
