// header file that is necessary to compile on Windows
#ifndef UNISTD_H
#define UNISTD_H

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define WIN32_LEAN_AND_MEAN

#شامل <stdio.h>
#شامل <math.h>
#شامل <time.h>
#شامل <stdlib.h> // for malloc and free
#شامل <string.h>
#شامل <direct.h> // for _mkdir and _stat
#شامل <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#pragma comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#شامل <winsock2.h>

#define CLOCK_MONOTONIC 0
ایستا درونخطی صحیح clock_gettime(صحیح ignore_variable, ساختار timespec* tv)
{
    بازگشت timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#define OMP /* turn it on */
#define F_OK 0
#define access _access

#define TURN_OFF_FP_FAST __pragma(float_control( precise, on, افزودن )) // Save current setting and turn on /fp:precise
#define TURN_ON_FP_FAST  __pragma(float_control(برداشت)) // Restore file's default settings

#define mkdir(path, mode) _mkdir(path) /* sketchy way to get mkdir to work on windows */
#define stat _stat

تعریفنوع ساختار glob_t {
    size_t gl_pathc;    // Count of matched pathnames
    نویسه **gl_pathv;    // List of matched pathnames
} glob_t;

ایستا درونخطی تهی replace_forward_slashes(نویسه* str) {
    وقتی (*str) {
        اگر (*str == '/') {
            *str = '\\';
        }
        str++;
    }
}

ایستا درونخطی تهی globfree(glob_t *pglob) {
    برای (size_t i = 0; i < pglob->gl_pathc; ++i) {
        آزاد(pglob->gl_pathv[i]); // Free the allocated memory for each filename
    }
    آزاد(pglob->gl_pathv); // Free the allocated memory for the list of filenames
}

ایستا درونخطی صحیح glob(ثابت نویسه* pattern, صحیح ignored_flags, صحیح (*ignored_errfunc)(ثابت نویسه* epath, صحیح eerrno), glob_t* pglob){
    ساختار _finddata_t find_file_data;
    نویسه full_path[576]; // stored in pglob->gl_pathv[n]
    نویسه directory_path[512] = {0}; // Store the directory path from the pattern
    نویسه pattern_copy[512]; // Copy of the pattern to modify

    strncpy_s(pattern_copy, اندازه(pattern_copy) - 1, pattern, اندازه(pattern_copy) - 1);

    replace_forward_slashes (pattern_copy); // Replace forward slashes with backslashes

    اگر (strchr(pattern_copy, '\\') != (تهی*) پوچ) {
        strncpy_s(directory_path, اندازه(directory_path) - 1, pattern_copy, strrchr(pattern_copy, '\\') - pattern_copy + 1);
        directory_path[strrchr(pattern_copy, '\\') - pattern_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t find_handle = _findfirst(pattern_copy, &find_file_data);

    اگر (find_handle == -1) {
        بازگشت 1; // No files found
    }

    size_t file_شمار = 0;
    size_t max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (نویسه **) malloc(max_files * اندازه(نویسه*)); // freed in globfree

    اگر (pglob->gl_pathv == پوچ) {
        _findclose(find_handle);
        بازگشت 2; // Memory allocation failed
    }

    انجامبده {
        اگر (file_شمار >= max_files) {
            _findclose(find_handle);
            بازگشت 2; // Too many files found
            }

        snprintf(full_path, اندازه(full_path), "%s%s", directory_path, find_file_data.name);

        pglob->gl_pathv[file_شمار] = _strdup(full_path); // freed in globfree

        اگر (pglob->gl_pathv[file_شمار] == پوچ) {
            _findclose(find_handle);
            بازگشت 2; // Memory allocation for filename failed
        }
        file_شمار++;
    } وقتی (_findnext(find_handle, &find_file_data) == 0);

    _findclose(find_handle);

    pglob->gl_pathc = file_شمار;
    بازگشت 0;
}

// dirent.h support

#define MAX_PATH_LENGTH 512
تعریفنوع ساختار dirent {
    نویسه d_name[MAX_PATH_LENGTH];
} dirent;

تعریفنوع ساختار DIR {
    intptr_t handle;
    ساختار _finddata_t findFileData;
    صحیح firstRead;
} DIR;

ایستا درونخطی DIR *opendir(ثابت نویسه *name) {
    DIR *dir = (DIR *)malloc(اندازه(DIR));
    اگر (dir == پوچ) {
        بازگشت پوچ;
    }

    نویسه جستجو_Path[MAX_PATH_LENGTH];

    snprintf(جستجو_Path, MAX_PATH_LENGTH, "%s\\*.*", name);

    dir->handle = _findfirst(جستجو_Path, &dir->findFileData);
    اگر (dir->handle == -1) {
        آزاد(dir);
        بازگشت پوچ;
    }

    dir->firstRead = 1;
    بازگشت dir;
}

ایستا درونخطی ساختار dirent *readdir(DIR *directory) {
    ایستا ساختار dirent result;

    اگر (directory->firstRead) {
        directory->firstRead = 0;
    } وگرنه {
        اگر (_findnext(directory->handle, &directory->findFileData) != 0) {
            بازگشت پوچ;
        }
    }

    strncpy(result.d_name, directory->findFileData.name, MAX_PATH_LENGTH);
    result.d_name[MAX_PATH_LENGTH - 1] = '\0'; // Ensure null termination
    بازگشت &result;
}

ایستا درونخطی صحیح closedir(DIR *directory) {
    اگر (directory == پوچ) {
        بازگشت -1;
    }

    اگر (_findclose(directory->handle) != 0) {
        بازگشت -1;
    }

    آزاد(directory);
    بازگشت 0;
}
#endif // UNISTD_H
