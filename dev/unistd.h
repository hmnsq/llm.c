// header file that is necessary to compile on Windows
#ifndef UNISTD_H
#define UNISTD_H

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define WIN32_LEAN_AND_MEAN

#அந்தர்பாவ <stdio.h>
#அந்தர்பாவ <math.h>
#அந்தர்பாவ <time.h>
#அந்தர்பாவ <stdlib.h> // for malloc and free
#அந்தர்பாவ <string.h>
#அந்தர்பாவ <direct.h> // for _mkdir and _stat
#அந்தர்பாவ <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#pragma comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#அந்தர்பாவ <winsock2.h>

#define CLOCK_MONOTONIC 0
நிலையான உள்வரி முழு clock_gettime(முழு ignore_variable, கட்டமைப்பு timespec* tv)
{
    திரும்பு timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#define OMP /* turn it on */
#define F_OK 0
#define access _access

#define TURN_OFF_FP_FAST __pragma(float_control( precise, on, தள்ளு )) // Save current setting and turn on /fp:precise
#define TURN_ON_FP_FAST  __pragma(float_control(எடு)) // Restore file's default settings

#define mkdir(path, mode) _mkdir(path) /* sketchy way to get mkdir to work on windows */
#define stat _stat

வகைவரை கட்டமைப்பு glob_t {
    size_t gl_pathc;    // Count of matched pathnames
    எழுத்து **gl_pathv;    // List of matched pathnames
} glob_t;

நிலையான உள்வரி வெறுமை replace_forward_slashes(எழுத்து* str) {
    போது (*str) {
        எனில் (*str == '/') {
            *str = '\\';
        }
        str++;
    }
}

நிலையான உள்வரி வெறுமை globfree(glob_t *pglob) {
    க்கு (size_t i = 0; i < pglob->gl_pathc; ++i) {
        விடுவி(pglob->gl_pathv[i]); // Free the allocated memory for each filename
    }
    விடுவி(pglob->gl_pathv); // Free the allocated memory for the list of filenames
}

நிலையான உள்வரி முழு glob(மாறிலி எழுத்து* pattern, முழு ignored_flags, முழு (*ignored_errfunc)(மாறிலி எழுத்து* epath, முழு eerrno), glob_t* pglob){
    கட்டமைப்பு _finddata_t find_file_data;
    எழுத்து full_path[576]; // stored in pglob->gl_pathv[n]
    எழுத்து directory_path[512] = {0}; // Store the directory path from the pattern
    எழுத்து pattern_copy[512]; // Copy of the pattern to modify

    strncpy_s(pattern_copy, அளவு(pattern_copy) - 1, pattern, அளவு(pattern_copy) - 1);

    replace_forward_slashes (pattern_copy); // Replace forward slashes with backslashes

    எனில் (strchr(pattern_copy, '\\') != (வெறுமை*) வெற்று) {
        strncpy_s(directory_path, அளவு(directory_path) - 1, pattern_copy, strrchr(pattern_copy, '\\') - pattern_copy + 1);
        directory_path[strrchr(pattern_copy, '\\') - pattern_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t find_handle = _findfirst(pattern_copy, &find_file_data);

    எனில் (find_handle == -1) {
        திரும்பு 1; // No files found
    }

    size_t file_மொத்தம் = 0;
    size_t max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (எழுத்து **) malloc(max_files * அளவு(எழுத்து*)); // freed in globfree

    எனில் (pglob->gl_pathv == வெற்று) {
        _findclose(find_handle);
        திரும்பு 2; // Memory allocation failed
    }

    செய் {
        எனில் (file_மொத்தம் >= max_files) {
            _findclose(find_handle);
            திரும்பு 2; // Too many files found
            }

        snprintf(full_path, அளவு(full_path), "%s%s", directory_path, find_file_data.name);

        pglob->gl_pathv[file_மொத்தம்] = _strdup(full_path); // freed in globfree

        எனில் (pglob->gl_pathv[file_மொத்தம்] == வெற்று) {
            _findclose(find_handle);
            திரும்பு 2; // Memory allocation for filename failed
        }
        file_மொத்தம்++;
    } போது (_findnext(find_handle, &find_file_data) == 0);

    _findclose(find_handle);

    pglob->gl_pathc = file_மொத்தம்;
    திரும்பு 0;
}

// dirent.h support

#define MAX_PATH_LENGTH 512
வகைவரை கட்டமைப்பு dirent {
    எழுத்து d_name[MAX_PATH_LENGTH];
} dirent;

வகைவரை கட்டமைப்பு DIR {
    intptr_t handle;
    கட்டமைப்பு _finddata_t findFileData;
    முழு firstRead;
} DIR;

நிலையான உள்வரி DIR *opendir(மாறிலி எழுத்து *name) {
    DIR *dir = (DIR *)malloc(அளவு(DIR));
    எனில் (dir == வெற்று) {
        திரும்பு வெற்று;
    }

    எழுத்து தேடல்_Path[MAX_PATH_LENGTH];

    snprintf(தேடல்_Path, MAX_PATH_LENGTH, "%s\\*.*", name);

    dir->handle = _findfirst(தேடல்_Path, &dir->findFileData);
    எனில் (dir->handle == -1) {
        விடுவி(dir);
        திரும்பு வெற்று;
    }

    dir->firstRead = 1;
    திரும்பு dir;
}

நிலையான உள்வரி கட்டமைப்பு dirent *readdir(DIR *directory) {
    நிலையான கட்டமைப்பு dirent result;

    எனில் (directory->firstRead) {
        directory->firstRead = 0;
    } இல்லையெனில் {
        எனில் (_findnext(directory->handle, &directory->findFileData) != 0) {
            திரும்பு வெற்று;
        }
    }

    strncpy(result.d_name, directory->findFileData.name, MAX_PATH_LENGTH);
    result.d_name[MAX_PATH_LENGTH - 1] = '\0'; // Ensure null termination
    திரும்பு &result;
}

நிலையான உள்வரி முழு closedir(DIR *directory) {
    எனில் (directory == வெற்று) {
        திரும்பு -1;
    }

    எனில் (_findclose(directory->handle) != 0) {
        திரும்பு -1;
    }

    விடுவி(directory);
    திரும்பு 0;
}
#endif // UNISTD_H
