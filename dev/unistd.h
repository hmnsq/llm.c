// header file that is necessary to compile on Windows
#که_نه_تعریف_شوی UNISTD_H
#تعریف_کړه UNISTD_H

#تعریف_کړه _CRT_SECURE_NO_WARNINGS
#تعریف_کړه USE_MATH_تعریفوي
#تعریف_کړه WIN32_LEAN_AND_منځنۍ

#ضمن <stdio.h>
#ضمن <math.h>
#ضمن <time.h>
#ضمن <stdlib.h> // for malloc and free
#ضمن <string.h>
#ضمن <direct.h> // for _mkdir and _stat
#ضمن <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#لارښوونه comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#ضمن <winsock2.h>

#تعریف_کړه CLOCK_MONOTONIC 0
ساكن مضمن صحيح clock_gettime(صحيح ignore_variable, بنية timespec* tv)
{
    أرجع timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#تعریف_کړه OMP /* turn it on */
#تعریف_کړه F_OK 0
#تعریف_کړه access _access

#تعریف_کړه TURN_OFF_FP_FAST __pragma(float_control( precise, on, ټېل )) // Save current setting and turn on /fp:precise
#تعریف_کړه TURN_ON_FP_FAST  __pragma(float_control(راوباسه)) // Restore file's default settings

#تعریف_کړه mkdir(لار, ډېرپېښ) _mkdir(لار) /* sketchy way to get mkdir to work on windows */
#تعریف_کړه stat _stat

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

ساكن مضمن صحيح glob(ثابت حرف* نمونه, صحيح ignored_flags, صحيح (*ignored_errfunc)(ثابت حرف* epath, صحيح eerrno), glob_t* pglob){
    بنية _finddata_t پیدا_کړه_file_معلومات;
    حرف ډک_لار[576]; // stored in pglob->gl_pathv[n]
    حرف directory_لار[512] = {0}; // Store the directory path from the pattern
    حرف نمونه_copy[512]; // Copy of the pattern to modify

    strncpy_s(نمونه_copy, حجم(نمونه_copy) - 1, نمونه, حجم(نمونه_copy) - 1);

    replace_forward_slashes (نمونه_copy); // Replace forward slashes with backslashes

    إذا (strchr(نمونه_copy, '\\') != (فراغ*) عدم) {
        strncpy_s(directory_لار, حجم(directory_لار) - 1, نمونه_copy, strrchr(نمونه_copy, '\\') - نمونه_copy + 1);
        directory_لار[strrchr(نمونه_copy, '\\') - نمونه_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t پیدا_کړه_handle = _findfirst(نمونه_copy, &پیدا_کړه_file_معلومات);

    إذا (پیدا_کړه_handle == -1) {
        أرجع 1; // No files found
    }

    size_t file_شمېر = 0;
    size_t max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (حرف **) حافظه_ورکړه(max_files * حجم(حرف*)); // freed in globfree

    إذا (pglob->gl_pathv == عدم) {
        _findclose(پیدا_کړه_handle);
        أرجع 2; // Memory allocation failed
    }

    افعل {
        إذا (file_شمېر >= max_files) {
            _findclose(پیدا_کړه_handle);
            أرجع 2; // Too many files found
            }

        snprintf(ډک_لار, حجم(ډک_لار), "%s%s", directory_لار, پیدا_کړه_file_معلومات.name);

        pglob->gl_pathv[file_شمېر] = _strdup(ډک_لار); // freed in globfree

        إذا (pglob->gl_pathv[file_شمېر] == عدم) {
            _findclose(پیدا_کړه_handle);
            أرجع 2; // Memory allocation for filename failed
        }
        file_شمېر++;
    } طالما (_findnext(پیدا_کړه_handle, &پیدا_کړه_file_معلومات) == 0);

    _findclose(پیدا_کړه_handle);

    pglob->gl_pathc = file_شمېر;
    أرجع 0;
}

// dirent.h support

#تعریف_کړه MAX_لار_اوږدتیا 512
تعريفنوع بنية dirent {
    حرف d_name[MAX_لار_اوږدتیا];
} dirent;

تعريفنوع بنية DIR {
    intptr_t handle;
    بنية _finddata_t پیدا_کړه_File_معلومات;
    صحيح لومړی_Read;
} DIR;

ساكن مضمن DIR *opendir(ثابت حرف *name) {
    DIR *dir = (DIR *)حافظه_ورکړه(حجم(DIR));
    إذا (dir == عدم) {
        أرجع عدم;
    }

    حرف پلټنه_لار[MAX_لار_اوږدتیا];

    snprintf(پلټنه_لار, MAX_لار_اوږدتیا, "%s\\*.*", name);

    dir->handle = _findfirst(پلټنه_لار, &dir->پیدا_کړه_File_معلومات);
    إذا (dir->handle == -1) {
        حرر(dir);
        أرجع عدم;
    }

    dir->لومړی_Read = 1;
    أرجع dir;
}

ساكن مضمن بنية dirent *readdir(DIR *directory) {
    ساكن بنية dirent پایله;

    إذا (directory->لومړی_Read) {
        directory->لومړی_Read = 0;
    } وإلا {
        إذا (_findnext(directory->handle, &directory->پیدا_کړه_File_معلومات) != 0) {
            أرجع عدم;
        }
    }

    strncpy(پایله.d_name, directory->پیدا_کړه_File_معلومات.name, MAX_لار_اوږدتیا);
    پایله.d_name[MAX_لار_اوږدتیا - 1] = '\0'; // Ensure null termination
    أرجع &پایله;
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
#د_که_پای // UNISTD_H
