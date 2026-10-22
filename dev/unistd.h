// header file that is necessary to compile on Windows
#ifndef UNISTD_H
#define UNISTD_H

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define WIN32_LEAN_AND_MEAN

#ضمن <stdio.h>
#ضمن <math.h>
#ضمن <time.h>
#ضمن <stdlib.h> // for malloc and free
#ضمن <string.h>
#ضمن <direct.h> // for _mkdir and _stat
#ضمن <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#pragma comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#ضمن <winsock2.h>

#define CLOCK_MONOTONIC 0
ساكن مضمن صحيح clock_gettime(صحيح ignore_variable, بنية timespec* tv)
{
    أرجع timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#define OMP /* turn it on */
#define F_OK 0
#define access _access

#define TURN_OFF_FP_FAST __pragma(float_control( precise, on, ڈالو )) // Save current setting and turn on /fp:precise
#define TURN_ON_FP_FAST  __pragma(float_control(نکالو)) // Restore file's default settings

#define mkdir(path, mode) _mkdir(path) /* sketchy way to get mkdir to work on windows */
#define stat _stat

تعريفنوع بنية glob_t {
    size_t gl_pathc;    // Count of matched pathnames
    حرف **gl_pathv;    // List of matched pathnames
} glob_t;

ساكن مضمن فراغ replace_forward_slashes(حرف* str) {
    طالما (*str) {
        إذا (*str == '/') {
            *str = '\\';
        }
        str++;
    }
}

ساكن مضمن فراغ globfree(glob_t *pglob) {
    لكل (size_t i = 0; i < pglob->gl_pathc; ++i) {
        حرر(pglob->gl_pathv[i]); // Free the allocated memory for each filename
    }
    حرر(pglob->gl_pathv); // Free the allocated memory for the list of filenames
}

ساكن مضمن صحيح glob(ثابت حرف* pattern, صحيح ignored_flags, صحيح (*ignored_errfunc)(ثابت حرف* epath, صحيح eerrno), glob_t* pglob){
    بنية _finddata_t find_file_data;
    حرف full_path[576]; // stored in pglob->gl_pathv[n]
    حرف directory_path[512] = {0}; // Store the directory path from the pattern
    حرف pattern_copy[512]; // Copy of the pattern to modify

    strncpy_s(pattern_copy, حجم(pattern_copy) - 1, pattern, حجم(pattern_copy) - 1);

    replace_forward_slashes (pattern_copy); // Replace forward slashes with backslashes

    إذا (strchr(pattern_copy, '\\') != (فراغ*) عدم) {
        strncpy_s(directory_path, حجم(directory_path) - 1, pattern_copy, strrchr(pattern_copy, '\\') - pattern_copy + 1);
        directory_path[strrchr(pattern_copy, '\\') - pattern_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t find_handle = _findfirst(pattern_copy, &find_file_data);

    إذا (find_handle == -1) {
        أرجع 1; // No files found
    }

    size_t file_گنتی = 0;
    size_t max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (حرف **) malloc(max_files * حجم(حرف*)); // freed in globfree

    إذا (pglob->gl_pathv == عدم) {
        _findclose(find_handle);
        أرجع 2; // Memory allocation failed
    }

    افعل {
        إذا (file_گنتی >= max_files) {
            _findclose(find_handle);
            أرجع 2; // Too many files found
            }

        snprintf(full_path, حجم(full_path), "%s%s", directory_path, find_file_data.name);

        pglob->gl_pathv[file_گنتی] = _strdup(full_path); // freed in globfree

        إذا (pglob->gl_pathv[file_گنتی] == عدم) {
            _findclose(find_handle);
            أرجع 2; // Memory allocation for filename failed
        }
        file_گنتی++;
    } طالما (_findnext(find_handle, &find_file_data) == 0);

    _findclose(find_handle);

    pglob->gl_pathc = file_گنتی;
    أرجع 0;
}

// dirent.h support

#define MAX_PATH_LENGTH 512
تعريفنوع بنية dirent {
    حرف d_name[MAX_PATH_LENGTH];
} dirent;

تعريفنوع بنية DIR {
    intptr_t handle;
    بنية _finddata_t findFileData;
    صحيح firstRead;
} DIR;

ساكن مضمن DIR *opendir(ثابت حرف *name) {
    DIR *dir = (DIR *)malloc(حجم(DIR));
    إذا (dir == عدم) {
        أرجع عدم;
    }

    حرف تلاش_Path[MAX_PATH_LENGTH];

    snprintf(تلاش_Path, MAX_PATH_LENGTH, "%s\\*.*", name);

    dir->handle = _findfirst(تلاش_Path, &dir->findFileData);
    إذا (dir->handle == -1) {
        حرر(dir);
        أرجع عدم;
    }

    dir->firstRead = 1;
    أرجع dir;
}

ساكن مضمن بنية dirent *readdir(DIR *directory) {
    ساكن بنية dirent result;

    إذا (directory->firstRead) {
        directory->firstRead = 0;
    } وإلا {
        إذا (_findnext(directory->handle, &directory->findFileData) != 0) {
            أرجع عدم;
        }
    }

    strncpy(result.d_name, directory->findFileData.name, MAX_PATH_LENGTH);
    result.d_name[MAX_PATH_LENGTH - 1] = '\0'; // Ensure null termination
    أرجع &result;
}

ساكن مضمن صحيح closedir(DIR *directory) {
    إذا (directory == عدم) {
        أرجع -1;
    }

    إذا (_findclose(directory->handle) != 0) {
        أرجع -1;
    }

    حرر(directory);
    أرجع 0;
}
#endif // UNISTD_H
