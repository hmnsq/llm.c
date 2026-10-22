// header file that is necessary to compile on Windows
#ifndef UNISTD_H
#define UNISTD_H

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define WIN32_LEAN_AND_MEAN

#অন্তর্ভাব <stdio.h>
#অন্তর্ভাব <math.h>
#অন্তর্ভাব <time.h>
#অন্তর্ভাব <stdlib.h> // for malloc and free
#অন্তর্ভাব <string.h>
#অন্তর্ভাব <direct.h> // for _mkdir and _stat
#অন্তর্ভাব <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#pragma comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#অন্তর্ভাব <winsock2.h>

#define CLOCK_MONOTONIC 0
স্থির অন্তঃরেখ পূর্ণসংখ্যা clock_gettime(পূর্ণসংখ্যা ignore_variable, কাঠামো timespec* tv)
{
    ফেরত timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#define OMP /* turn it on */
#define F_OK 0
#define access _access

#define TURN_OFF_FP_FAST __pragma(float_control( precise, on, ঠেলো )) // Save current setting and turn on /fp:precise
#define TURN_ON_FP_FAST  __pragma(float_control(বের_করো)) // Restore file's default settings

#define mkdir(path, mode) _mkdir(path) /* sketchy way to get mkdir to work on windows */
#define stat _stat

প্রকারলেখ কাঠামো glob_t {
    size_t gl_pathc;    // Count of matched pathnames
    অক্ষর **gl_pathv;    // List of matched pathnames
} glob_t;

স্থির অন্তঃরেখ শূন্য replace_forward_slashes(অক্ষর* str) {
    যখন (*str) {
        যদি (*str == '/') {
            *str = '\\';
        }
        str++;
    }
}

স্থির অন্তঃরেখ শূন্য globfree(glob_t *pglob) {
    জন্য (size_t i = 0; i < pglob->gl_pathc; ++i) {
        মুক্ত(pglob->gl_pathv[i]); // Free the allocated memory for each filename
    }
    মুক্ত(pglob->gl_pathv); // Free the allocated memory for the list of filenames
}

স্থির অন্তঃরেখ পূর্ণসংখ্যা glob(ধ্রুবক অক্ষর* pattern, পূর্ণসংখ্যা ignored_flags, পূর্ণসংখ্যা (*ignored_errfunc)(ধ্রুবক অক্ষর* epath, পূর্ণসংখ্যা eerrno), glob_t* pglob){
    কাঠামো _finddata_t find_file_তথ্য;
    অক্ষর full_path[576]; // stored in pglob->gl_pathv[n]
    অক্ষর directory_path[512] = {0}; // Store the directory path from the pattern
    অক্ষর pattern_copy[512]; // Copy of the pattern to modify

    strncpy_s(pattern_copy, আকার(pattern_copy) - 1, pattern, আকার(pattern_copy) - 1);

    replace_forward_slashes (pattern_copy); // Replace forward slashes with backslashes

    যদি (strchr(pattern_copy, '\\') != (শূন্য*) ফাঁকা) {
        strncpy_s(directory_path, আকার(directory_path) - 1, pattern_copy, strrchr(pattern_copy, '\\') - pattern_copy + 1);
        directory_path[strrchr(pattern_copy, '\\') - pattern_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t find_handle = _findfirst(pattern_copy, &find_file_তথ্য);

    যদি (find_handle == -1) {
        ফেরত 1; // No files found
    }

    size_t file_গুনতি = 0;
    size_t max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (অক্ষর **) malloc(max_files * আকার(অক্ষর*)); // freed in globfree

    যদি (pglob->gl_pathv == ফাঁকা) {
        _findclose(find_handle);
        ফেরত 2; // Memory allocation failed
    }

    করো {
        যদি (file_গুনতি >= max_files) {
            _findclose(find_handle);
            ফেরত 2; // Too many files found
            }

        snprintf(full_path, আকার(full_path), "%s%s", directory_path, find_file_তথ্য.name);

        pglob->gl_pathv[file_গুনতি] = _strdup(full_path); // freed in globfree

        যদি (pglob->gl_pathv[file_গুনতি] == ফাঁকা) {
            _findclose(find_handle);
            ফেরত 2; // Memory allocation for filename failed
        }
        file_গুনতি++;
    } যখন (_findnext(find_handle, &find_file_তথ্য) == 0);

    _findclose(find_handle);

    pglob->gl_pathc = file_গুনতি;
    ফেরত 0;
}

// dirent.h support

#define MAX_PATH_দৈর্ঘ্য 512
প্রকারলেখ কাঠামো dirent {
    অক্ষর d_name[MAX_PATH_দৈর্ঘ্য];
} dirent;

প্রকারলেখ কাঠামো DIR {
    intptr_t handle;
    কাঠামো _finddata_t find_File_তথ্য;
    পূর্ণসংখ্যা firstRead;
} DIR;

স্থির অন্তঃরেখ DIR *opendir(ধ্রুবক অক্ষর *name) {
    DIR *dir = (DIR *)malloc(আকার(DIR));
    যদি (dir == ফাঁকা) {
        ফেরত ফাঁকা;
    }

    অক্ষর অনুসন্ধান_Path[MAX_PATH_দৈর্ঘ্য];

    snprintf(অনুসন্ধান_Path, MAX_PATH_দৈর্ঘ্য, "%s\\*.*", name);

    dir->handle = _findfirst(অনুসন্ধান_Path, &dir->find_File_তথ্য);
    যদি (dir->handle == -1) {
        মুক্ত(dir);
        ফেরত ফাঁকা;
    }

    dir->firstRead = 1;
    ফেরত dir;
}

স্থির অন্তঃরেখ কাঠামো dirent *readdir(DIR *directory) {
    স্থির কাঠামো dirent ফলাফল;

    যদি (directory->firstRead) {
        directory->firstRead = 0;
    } অন্যথা {
        যদি (_findnext(directory->handle, &directory->find_File_তথ্য) != 0) {
            ফেরত ফাঁকা;
        }
    }

    strncpy(ফলাফল.d_name, directory->find_File_তথ্য.name, MAX_PATH_দৈর্ঘ্য);
    ফলাফল.d_name[MAX_PATH_দৈর্ঘ্য - 1] = '\0'; // Ensure null termination
    ফেরত &ফলাফল;
}

স্থির অন্তঃরেখ পূর্ণসংখ্যা closedir(DIR *directory) {
    যদি (directory == ফাঁকা) {
        ফেরত -1;
    }

    যদি (_findclose(directory->handle) != 0) {
        ফেরত -1;
    }

    মুক্ত(directory);
    ফেরত 0;
}
#endif // UNISTD_H
