// header file that is necessary to compile on Windows
#اگر_تعریف_نشده UNISTD_H
#تعریف_کن UNISTD_H

#تعریف_کن _CRT_SECURE_NO_WARNINGS
#تعریف_کن USE_MATH_تعریف_میکند
#تعریف_کن WIN32_LEAN_AND_میانگین

#ضمن <stdio.h>
#ضمن <math.h>
#ضمن <time.h>
#ضمن <stdlib.h> // for malloc and free
#ضمن <string.h>
#ضمن <direct.h> // for _mkdir and _stat
#ضمن <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#رهنمود comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#ضمن <winsock2.h>

#تعریف_کن CLOCK_MONOTONIC 0
ساكن مضمن صحيح clock_gettime(صحيح ignore_variable, بنية timespec* tv)
{
    أرجع timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#تعریف_کن OMP /* turn it on */
#تعریف_کن F_OK 0
#تعریف_کن access _access

#تعریف_کن TURN_OFF_FP_FAST __pragma(float_control( precise, on, افزودن )) // Save current setting and turn on /fp:precise
#تعریف_کن TURN_ON_FP_FAST  __pragma(float_control(برداشت)) // Restore file's default settings

#تعریف_کن mkdir(مسیر, نما) _mkdir(مسیر) /* sketchy way to get mkdir to work on windows */
#تعریف_کن stat _stat

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

ساكن مضمن صحيح glob(ثابت حرف* الگو, صحيح ignored_flags, صحيح (*ignored_errfunc)(ثابت حرف* epath, صحيح eerrno), glob_t* pglob){
    بنية _finddata_t پیدا_کن_file_داده;
    حرف پر_مسیر[576]; // stored in pglob->gl_pathv[n]
    حرف directory_مسیر[512] = {0}; // Store the directory path from the pattern
    حرف الگو_copy[512]; // Copy of the pattern to modify

    strncpy_s(الگو_copy, حجم(الگو_copy) - 1, الگو, حجم(الگو_copy) - 1);

    replace_forward_slashes (الگو_copy); // Replace forward slashes with backslashes

    إذا (strchr(الگو_copy, '\\') != (فراغ*) عدم) {
        strncpy_s(directory_مسیر, حجم(directory_مسیر) - 1, الگو_copy, strrchr(الگو_copy, '\\') - الگو_copy + 1);
        directory_مسیر[strrchr(الگو_copy, '\\') - الگو_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t پیدا_کن_handle = _findfirst(الگو_copy, &پیدا_کن_file_داده);

    إذا (پیدا_کن_handle == -1) {
        أرجع 1; // No files found
    }

    size_t file_شمار = 0;
    size_t max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (حرف **) حافظه_بده(max_files * حجم(حرف*)); // freed in globfree

    إذا (pglob->gl_pathv == عدم) {
        _findclose(پیدا_کن_handle);
        أرجع 2; // Memory allocation failed
    }

    افعل {
        إذا (file_شمار >= max_files) {
            _findclose(پیدا_کن_handle);
            أرجع 2; // Too many files found
            }

        snprintf(پر_مسیر, حجم(پر_مسیر), "%s%s", directory_مسیر, پیدا_کن_file_داده.name);

        pglob->gl_pathv[file_شمار] = _strdup(پر_مسیر); // freed in globfree

        إذا (pglob->gl_pathv[file_شمار] == عدم) {
            _findclose(پیدا_کن_handle);
            أرجع 2; // Memory allocation for filename failed
        }
        file_شمار++;
    } طالما (_findnext(پیدا_کن_handle, &پیدا_کن_file_داده) == 0);

    _findclose(پیدا_کن_handle);

    pglob->gl_pathc = file_شمار;
    أرجع 0;
}

// dirent.h support

#تعریف_کن MAX_مسیر_درازی 512
تعريفنوع بنية dirent {
    حرف d_name[MAX_مسیر_درازی];
} dirent;

تعريفنوع بنية DIR {
    intptr_t handle;
    بنية _finddata_t پیدا_کن_File_داده;
    صحيح اولی_Read;
} DIR;

ساكن مضمن DIR *opendir(ثابت حرف *name) {
    DIR *dir = (DIR *)حافظه_بده(حجم(DIR));
    إذا (dir == عدم) {
        أرجع عدم;
    }

    حرف جستجو_مسیر[MAX_مسیر_درازی];

    snprintf(جستجو_مسیر, MAX_مسیر_درازی, "%s\\*.*", name);

    dir->handle = _findfirst(جستجو_مسیر, &dir->پیدا_کن_File_داده);
    إذا (dir->handle == -1) {
        حرر(dir);
        أرجع عدم;
    }

    dir->اولی_Read = 1;
    أرجع dir;
}

ساكن مضمن بنية dirent *readdir(DIR *directory) {
    ساكن بنية dirent نتیجه;

    إذا (directory->اولی_Read) {
        directory->اولی_Read = 0;
    } وإلا {
        إذا (_findnext(directory->handle, &directory->پیدا_کن_File_داده) != 0) {
            أرجع عدم;
        }
    }

    strncpy(نتیجه.d_name, directory->پیدا_کن_File_داده.name, MAX_مسیر_درازی);
    نتیجه.d_name[MAX_مسیر_درازی - 1] = '\0'; // Ensure null termination
    أرجع &نتیجه;
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
#پایان_اگر // UNISTD_H
