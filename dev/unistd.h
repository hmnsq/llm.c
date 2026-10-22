// header file that is necessary to compile on Windows
#ifndef UNISTD_H
#define UNISTD_H

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define WIN32_LEAN_AND_MEAN

#അന്തര്ഭാവ <stdio.h>
#അന്തര്ഭാവ <math.h>
#അന്തര്ഭാവ <time.h>
#അന്തര്ഭാവ <stdlib.h> // for malloc and free
#അന്തര്ഭാവ <string.h>
#അന്തര്ഭാവ <direct.h> // for _mkdir and _stat
#അന്തര്ഭാവ <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#pragma comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#അന്തര്ഭാവ <winsock2.h>

#define CLOCK_MONOTONIC 0
സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ clock_gettime(പൂർണ്ണസംഖ്യ ignore_variable, ഘടന timespec* tv)
{
    തിരികെ timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#define OMP /* turn it on */
#define F_OK 0
#define access _access

#define TURN_OFF_FP_FAST __pragma(float_control( precise, on, തള്ളുക )) // Save current setting and turn on /fp:precise
#define TURN_ON_FP_FAST  __pragma(float_control(എടുക്കുക)) // Restore file's default settings

#define mkdir(path, mode) _mkdir(path) /* sketchy way to get mkdir to work on windows */
#define stat _stat

തരനിർവചനം ഘടന glob_t {
    size_t gl_pathc;    // Count of matched pathnames
    അക്ഷരം **gl_pathv;    // List of matched pathnames
} glob_t;

സ്ഥിതം വരിയിൽ ശൂന്യം replace_forward_slashes(അക്ഷരം* str) {
    ആകുമ്പോൾ (*str) {
        എങ്കിൽ (*str == '/') {
            *str = '\\';
        }
        str++;
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം globfree(glob_t *pglob) {
    വേണ്ടി (size_t i = 0; i < pglob->gl_pathc; ++i) {
        സ്വതന്ത്രം(pglob->gl_pathv[i]); // Free the allocated memory for each filename
    }
    സ്വതന്ത്രം(pglob->gl_pathv); // Free the allocated memory for the list of filenames
}

സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ glob(സ്ഥിരം അക്ഷരം* pattern, പൂർണ്ണസംഖ്യ ignored_flags, പൂർണ്ണസംഖ്യ (*ignored_errfunc)(സ്ഥിരം അക്ഷരം* epath, പൂർണ്ണസംഖ്യ eerrno), glob_t* pglob){
    ഘടന _finddata_t find_file_data;
    അക്ഷരം full_path[576]; // stored in pglob->gl_pathv[n]
    അക്ഷരം directory_path[512] = {0}; // Store the directory path from the pattern
    അക്ഷരം pattern_copy[512]; // Copy of the pattern to modify

    strncpy_s(pattern_copy, വലുപ്പം(pattern_copy) - 1, pattern, വലുപ്പം(pattern_copy) - 1);

    replace_forward_slashes (pattern_copy); // Replace forward slashes with backslashes

    എങ്കിൽ (strchr(pattern_copy, '\\') != (ശൂന്യം*) ശൂന്യ) {
        strncpy_s(directory_path, വലുപ്പം(directory_path) - 1, pattern_copy, strrchr(pattern_copy, '\\') - pattern_copy + 1);
        directory_path[strrchr(pattern_copy, '\\') - pattern_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t find_handle = _findfirst(pattern_copy, &find_file_data);

    എങ്കിൽ (find_handle == -1) {
        തിരികെ 1; // No files found
    }

    size_t file_എണ്ണം = 0;
    size_t max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (അക്ഷരം **) malloc(max_files * വലുപ്പം(അക്ഷരം*)); // freed in globfree

    എങ്കിൽ (pglob->gl_pathv == ശൂന്യ) {
        _findclose(find_handle);
        തിരികെ 2; // Memory allocation failed
    }

    ചെയ്യുക {
        എങ്കിൽ (file_എണ്ണം >= max_files) {
            _findclose(find_handle);
            തിരികെ 2; // Too many files found
            }

        snprintf(full_path, വലുപ്പം(full_path), "%s%s", directory_path, find_file_data.name);

        pglob->gl_pathv[file_എണ്ണം] = _strdup(full_path); // freed in globfree

        എങ്കിൽ (pglob->gl_pathv[file_എണ്ണം] == ശൂന്യ) {
            _findclose(find_handle);
            തിരികെ 2; // Memory allocation for filename failed
        }
        file_എണ്ണം++;
    } ആകുമ്പോൾ (_findnext(find_handle, &find_file_data) == 0);

    _findclose(find_handle);

    pglob->gl_pathc = file_എണ്ണം;
    തിരികെ 0;
}

// dirent.h support

#define MAX_PATH_LENGTH 512
തരനിർവചനം ഘടന dirent {
    അക്ഷരം d_name[MAX_PATH_LENGTH];
} dirent;

തരനിർവചനം ഘടന DIR {
    intptr_t handle;
    ഘടന _finddata_t findFileData;
    പൂർണ്ണസംഖ്യ firstRead;
} DIR;

സ്ഥിതം വരിയിൽ DIR *opendir(സ്ഥിരം അക്ഷരം *name) {
    DIR *dir = (DIR *)malloc(വലുപ്പം(DIR));
    എങ്കിൽ (dir == ശൂന്യ) {
        തിരികെ ശൂന്യ;
    }

    അക്ഷരം തിരയൽ_Path[MAX_PATH_LENGTH];

    snprintf(തിരയൽ_Path, MAX_PATH_LENGTH, "%s\\*.*", name);

    dir->handle = _findfirst(തിരയൽ_Path, &dir->findFileData);
    എങ്കിൽ (dir->handle == -1) {
        സ്വതന്ത്രം(dir);
        തിരികെ ശൂന്യ;
    }

    dir->firstRead = 1;
    തിരികെ dir;
}

സ്ഥിതം വരിയിൽ ഘടന dirent *readdir(DIR *directory) {
    സ്ഥിതം ഘടന dirent result;

    എങ്കിൽ (directory->firstRead) {
        directory->firstRead = 0;
    } അല്ലെങ്കിൽ {
        എങ്കിൽ (_findnext(directory->handle, &directory->findFileData) != 0) {
            തിരികെ ശൂന്യ;
        }
    }

    strncpy(result.d_name, directory->findFileData.name, MAX_PATH_LENGTH);
    result.d_name[MAX_PATH_LENGTH - 1] = '\0'; // Ensure null termination
    തിരികെ &result;
}

സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ closedir(DIR *directory) {
    എങ്കിൽ (directory == ശൂന്യ) {
        തിരികെ -1;
    }

    എങ്കിൽ (_findclose(directory->handle) != 0) {
        തിരികെ -1;
    }

    സ്വതന്ത്രം(directory);
    തിരികെ 0;
}
#endif // UNISTD_H
