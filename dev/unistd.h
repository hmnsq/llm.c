// header file that is necessary to compile on Windows
#اگر_غیر_تعریف_شدہ UNISTD_H
#تعریف_کرو UNISTD_H

#تعریف_کرو _CRT_SECURE_NO_WARNINGS
#تعریف_کرو USE_MATH_متعین_کرتا
#تعریف_کرو WIN32_LEAN_AND_اوسط

#شامل <stdio.h>
#شامل <math.h>
#شامل <time.h>
#شامل <stdlib.h> // for malloc and free
#شامل <string.h>
#شامل <direct.h> // for _mkdir and _stat
#شامل <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#ہدایت comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#شامل <winsock2.h>

#تعریف_کرو CLOCK_MONOTONIC 0
ساکن اندرونی صحیح clock_gettime(صحیح ignore_variable, ساخت timespec* tv)
{
    واپس timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#تعریف_کرو OMP /* turn it on */
#تعریف_کرو F_OK 0
#تعریف_کرو access _access

#تعریف_کرو TURN_OFF_FP_FAST __pragma(float_control( precise, on, ڈالو )) // Save current setting and turn on /fp:precise
#تعریف_کرو TURN_ON_FP_FAST  __pragma(float_control(نکالو)) // Restore file's default settings

#تعریف_کرو mkdir(راستہ, منوال) _mkdir(راستہ) /* sketchy way to get mkdir to work on windows */
#تعریف_کرو stat _stat

قسمتعریف ساخت glob_t {
    size_t gl_pathc;    // Count of matched pathnames
    حرف **gl_pathv;    // List of matched pathnames
} glob_t;

ساکن اندرونی خالی replace_forward_slashes(حرف* str) {
    جب (*str) {
        اگر (*str == '/') {
            *str = '\\';
        }
        str++;
    }
}

ساکن اندرونی خالی globfree(glob_t *pglob) {
    برائے (size_t i = 0; i < pglob->gl_pathc; ++i) {
        آزاد(pglob->gl_pathv[i]); // Free the allocated memory for each filename
    }
    آزاد(pglob->gl_pathv); // Free the allocated memory for the list of filenames
}

ساکن اندرونی صحیح glob(ثابت حرف* نمونہ, صحیح ignored_flags, صحیح (*ignored_errfunc)(ثابت حرف* epath, صحیح eerrno), glob_t* pglob){
    ساخت _finddata_t ڈھونڈو_file_ڈیٹا;
    حرف بھرا_راستہ[576]; // stored in pglob->gl_pathv[n]
    حرف directory_راستہ[512] = {0}; // Store the directory path from the pattern
    حرف نمونہ_copy[512]; // Copy of the pattern to modify

    strncpy_s(نمونہ_copy, حجم(نمونہ_copy) - 1, نمونہ, حجم(نمونہ_copy) - 1);

    replace_forward_slashes (نمونہ_copy); // Replace forward slashes with backslashes

    اگر (strchr(نمونہ_copy, '\\') != (خالی*) تہی) {
        strncpy_s(directory_راستہ, حجم(directory_راستہ) - 1, نمونہ_copy, strrchr(نمونہ_copy, '\\') - نمونہ_copy + 1);
        directory_راستہ[strrchr(نمونہ_copy, '\\') - نمونہ_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t ڈھونڈو_handle = _findfirst(نمونہ_copy, &ڈھونڈو_file_ڈیٹا);

    اگر (ڈھونڈو_handle == -1) {
        واپس 1; // No files found
    }

    size_t file_گنتی = 0;
    size_t max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (حرف **) حافظہ_دو(max_files * حجم(حرف*)); // freed in globfree

    اگر (pglob->gl_pathv == تہی) {
        _findclose(ڈھونڈو_handle);
        واپس 2; // Memory allocation failed
    }

    کرو {
        اگر (file_گنتی >= max_files) {
            _findclose(ڈھونڈو_handle);
            واپس 2; // Too many files found
            }

        snprintf(بھرا_راستہ, حجم(بھرا_راستہ), "%s%s", directory_راستہ, ڈھونڈو_file_ڈیٹا.name);

        pglob->gl_pathv[file_گنتی] = _strdup(بھرا_راستہ); // freed in globfree

        اگر (pglob->gl_pathv[file_گنتی] == تہی) {
            _findclose(ڈھونڈو_handle);
            واپس 2; // Memory allocation for filename failed
        }
        file_گنتی++;
    } جب (_findnext(ڈھونڈو_handle, &ڈھونڈو_file_ڈیٹا) == 0);

    _findclose(ڈھونڈو_handle);

    pglob->gl_pathc = file_گنتی;
    واپس 0;
}

// dirent.h support

#تعریف_کرو MAX_راستہ_لمبائی 512
قسمتعریف ساخت dirent {
    حرف d_name[MAX_راستہ_لمبائی];
} dirent;

قسمتعریف ساخت DIR {
    intptr_t handle;
    ساخت _finddata_t ڈھونڈو_File_ڈیٹا;
    صحیح پہلا_Read;
} DIR;

ساکن اندرونی DIR *opendir(ثابت حرف *name) {
    DIR *dir = (DIR *)حافظہ_دو(حجم(DIR));
    اگر (dir == تہی) {
        واپس تہی;
    }

    حرف تلاش_راستہ[MAX_راستہ_لمبائی];

    snprintf(تلاش_راستہ, MAX_راستہ_لمبائی, "%s\\*.*", name);

    dir->handle = _findfirst(تلاش_راستہ, &dir->ڈھونڈو_File_ڈیٹا);
    اگر (dir->handle == -1) {
        آزاد(dir);
        واپس تہی;
    }

    dir->پہلا_Read = 1;
    واپس dir;
}

ساکن اندرونی ساخت dirent *readdir(DIR *directory) {
    ساکن ساخت dirent نتیجہ;

    اگر (directory->پہلا_Read) {
        directory->پہلا_Read = 0;
    } ورنہ {
        اگر (_findnext(directory->handle, &directory->ڈھونڈو_File_ڈیٹا) != 0) {
            واپس تہی;
        }
    }

    strncpy(نتیجہ.d_name, directory->ڈھونڈو_File_ڈیٹا.name, MAX_راستہ_لمبائی);
    نتیجہ.d_name[MAX_راستہ_لمبائی - 1] = '\0'; // Ensure null termination
    واپس &نتیجہ;
}

ساکن اندرونی صحیح closedir(DIR *directory) {
    اگر (directory == تہی) {
        واپس -1;
    }

    اگر (_findclose(directory->handle) != 0) {
        واپس -1;
    }

    آزاد(directory);
    واپس 0;
}
#اگر_ختم // UNISTD_H
