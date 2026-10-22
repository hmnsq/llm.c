// header file that is necessary to compile on Windows
#ifndef UNISTD_H
#define UNISTD_H

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define WIN32_LEAN_AND_MEAN

#অন্তর্ভাৱ <stdio.h>
#অন্তর্ভাৱ <math.h>
#অন্তর্ভাৱ <time.h>
#অন্তর্ভাৱ <stdlib.h> // for malloc and free
#অন্তর্ভাৱ <string.h>
#অন্তর্ভাৱ <direct.h> // for _mkdir and _stat
#অন্তর্ভাৱ <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#pragma comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#অন্তর্ভাৱ <winsock2.h>

#define CLOCK_MONOTONIC 0
স্থির অন্তঃস্থ পূর্ণাঙ্ক clock_gettime(পূর্ণাঙ্ক ignore_variable, সংরচনা timespec* tv)
{
    ঘূৰাই timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#define OMP /* turn it on */
#define F_OK 0
#define access _access

#define TURN_OFF_FP_FAST __pragma(float_control( precise, on, ঠেলো )) // Save current setting and turn on /fp:precise
#define TURN_ON_FP_FAST  __pragma(float_control(বেৰ_কৰো)) // Restore file's default settings

#define mkdir(path, mode) _mkdir(path) /* sketchy way to get mkdir to work on windows */
#define stat _stat

প্রকারনির্দেশ সংরচনা glob_t {
    size_t gl_pathc;    // Count of matched pathnames
    ৱর্ণ **gl_pathv;    // List of matched pathnames
} glob_t;

স্থির অন্তঃস্থ শূন্য replace_forward_slashes(ৱর্ণ* str) {
    যেতিয়ালৈ (*str) {
        যদি (*str == '/') {
            *str = '\\';
        }
        str++;
    }
}

স্থির অন্তঃস্থ শূন্য globfree(glob_t *pglob) {
    বাবে (size_t i = 0; i < pglob->gl_pathc; ++i) {
        মুক্ত(pglob->gl_pathv[i]); // Free the allocated memory for each filename
    }
    মুক্ত(pglob->gl_pathv); // Free the allocated memory for the list of filenames
}

স্থির অন্তঃস্থ পূর্ণাঙ্ক glob(ধ্রুৱ ৱর্ণ* pattern, পূর্ণাঙ্ক ignored_flags, পূর্ণাঙ্ক (*ignored_errfunc)(ধ্রুৱ ৱর্ণ* epath, পূর্ণাঙ্ক eerrno), glob_t* pglob){
    সংরচনা _finddata_t find_file_data;
    ৱর্ণ full_path[576]; // stored in pglob->gl_pathv[n]
    ৱর্ণ directory_path[512] = {0}; // Store the directory path from the pattern
    ৱর্ণ pattern_copy[512]; // Copy of the pattern to modify

    strncpy_s(pattern_copy, পরিমাণ(pattern_copy) - 1, pattern, পরিমাণ(pattern_copy) - 1);

    replace_forward_slashes (pattern_copy); // Replace forward slashes with backslashes

    যদি (strchr(pattern_copy, '\\') != (শূন্য*) রিক্ত) {
        strncpy_s(directory_path, পরিমাণ(directory_path) - 1, pattern_copy, strrchr(pattern_copy, '\\') - pattern_copy + 1);
        directory_path[strrchr(pattern_copy, '\\') - pattern_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t find_handle = _findfirst(pattern_copy, &find_file_data);

    যদি (find_handle == -1) {
        ঘূৰাই 1; // No files found
    }

    size_t file_গুনতি = 0;
    size_t max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (ৱর্ণ **) malloc(max_files * পরিমাণ(ৱর্ণ*)); // freed in globfree

    যদি (pglob->gl_pathv == রিক্ত) {
        _findclose(find_handle);
        ঘূৰাই 2; // Memory allocation failed
    }

    কৰা {
        যদি (file_গুনতি >= max_files) {
            _findclose(find_handle);
            ঘূৰাই 2; // Too many files found
            }

        snprintf(full_path, পরিমাণ(full_path), "%s%s", directory_path, find_file_data.name);

        pglob->gl_pathv[file_গুনতি] = _strdup(full_path); // freed in globfree

        যদি (pglob->gl_pathv[file_গুনতি] == রিক্ত) {
            _findclose(find_handle);
            ঘূৰাই 2; // Memory allocation for filename failed
        }
        file_গুনতি++;
    } যেতিয়ালৈ (_findnext(find_handle, &find_file_data) == 0);

    _findclose(find_handle);

    pglob->gl_pathc = file_গুনতি;
    ঘূৰাই 0;
}

// dirent.h support

#define MAX_PATH_LENGTH 512
প্রকারনির্দেশ সংরচনা dirent {
    ৱর্ণ d_name[MAX_PATH_LENGTH];
} dirent;

প্রকারনির্দেশ সংরচনা DIR {
    intptr_t handle;
    সংরচনা _finddata_t findFileData;
    পূর্ণাঙ্ক firstRead;
} DIR;

স্থির অন্তঃস্থ DIR *opendir(ধ্রুৱ ৱর্ণ *name) {
    DIR *dir = (DIR *)malloc(পরিমাণ(DIR));
    যদি (dir == রিক্ত) {
        ঘূৰাই রিক্ত;
    }

    ৱর্ণ অনুসন্ধান_Path[MAX_PATH_LENGTH];

    snprintf(অনুসন্ধান_Path, MAX_PATH_LENGTH, "%s\\*.*", name);

    dir->handle = _findfirst(অনুসন্ধান_Path, &dir->findFileData);
    যদি (dir->handle == -1) {
        মুক্ত(dir);
        ঘূৰাই রিক্ত;
    }

    dir->firstRead = 1;
    ঘূৰাই dir;
}

স্থির অন্তঃস্থ সংরচনা dirent *readdir(DIR *directory) {
    স্থির সংরচনা dirent result;

    যদি (directory->firstRead) {
        directory->firstRead = 0;
    } অন্যথা {
        যদি (_findnext(directory->handle, &directory->findFileData) != 0) {
            ঘূৰাই রিক্ত;
        }
    }

    strncpy(result.d_name, directory->findFileData.name, MAX_PATH_LENGTH);
    result.d_name[MAX_PATH_LENGTH - 1] = '\0'; // Ensure null termination
    ঘূৰাই &result;
}

স্থির অন্তঃস্থ পূর্ণাঙ্ক closedir(DIR *directory) {
    যদি (directory == রিক্ত) {
        ঘূৰাই -1;
    }

    যদি (_findclose(directory->handle) != 0) {
        ঘূৰাই -1;
    }

    মুক্ত(directory);
    ঘূৰাই 0;
}
#endif // UNISTD_H
