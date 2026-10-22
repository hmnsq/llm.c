// header file that is necessary to compile on Windows
#جے_بنا_تعریف UNISTD_H
#تعریف_کرو UNISTD_H

#تعریف_کرو _CRT_SECURE_NO_WARNINGS
#تعریف_کرو USE_MATH_مقرر_کردا
#تعریف_کرو WIN32_LEAN_AND_اوسط

#ضمن <stdio.h>
#ضمن <math.h>
#ضمن <time.h>
#ضمن <stdlib.h> // for malloc and free
#ضمن <string.h>
#ضمن <direct.h> // for _mkdir and _stat
#ضمن <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#ہدایت comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#ضمن <winsock2.h>

#تعریف_کرو CLOCK_MONOTONIC 0
ساكن مضمن صحيح clock_gettime(صحيح ignore_variable, بنية timespec* tv)
{
    أرجع timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#تعریف_کرو OMP /* turn it on */
#تعریف_کرو F_OK 0
#تعریف_کرو access _access

#تعریف_کرو TURN_OFF_FP_FAST __pragma(float_control( precise, on, ڈالو )) // Save current setting and turn on /fp:precise
#تعریف_کرو TURN_ON_FP_FAST  __pragma(float_control(نکالو)) // Restore file's default settings

#تعریف_کرو mkdir(رستہ, منوال) _mkdir(رستہ) /* sketchy way to get mkdir to work on windows */
#تعریف_کرو stat _stat

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

ساكن مضمن صحيح glob(ثابت حرف* نمونہ, صحيح ignored_flags, صحيح (*ignored_errfunc)(ثابت حرف* epath, صحيح eerrno), glob_t* pglob){
    بنية _finddata_t لبھو_file_انکڑے;
    حرف بھریا_رستہ[576]; // stored in pglob->gl_pathv[n]
    حرف directory_رستہ[512] = {0}; // Store the directory path from the pattern
    حرف نمونہ_copy[512]; // Copy of the pattern to modify

    strncpy_s(نمونہ_copy, حجم(نمونہ_copy) - 1, نمونہ, حجم(نمونہ_copy) - 1);

    replace_forward_slashes (نمونہ_copy); // Replace forward slashes with backslashes

    إذا (strchr(نمونہ_copy, '\\') != (فراغ*) عدم) {
        strncpy_s(directory_رستہ, حجم(directory_رستہ) - 1, نمونہ_copy, strrchr(نمونہ_copy, '\\') - نمونہ_copy + 1);
        directory_رستہ[strrchr(نمونہ_copy, '\\') - نمونہ_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t لبھو_handle = _findfirst(نمونہ_copy, &لبھو_file_انکڑے);

    إذا (لبھو_handle == -1) {
        أرجع 1; // No files found
    }

    size_t file_گنتی = 0;
    size_t max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (حرف **) حافظہ_دیو(max_files * حجم(حرف*)); // freed in globfree

    إذا (pglob->gl_pathv == عدم) {
        _findclose(لبھو_handle);
        أرجع 2; // Memory allocation failed
    }

    افعل {
        إذا (file_گنتی >= max_files) {
            _findclose(لبھو_handle);
            أرجع 2; // Too many files found
            }

        snprintf(بھریا_رستہ, حجم(بھریا_رستہ), "%s%s", directory_رستہ, لبھو_file_انکڑے.name);

        pglob->gl_pathv[file_گنتی] = _strdup(بھریا_رستہ); // freed in globfree

        إذا (pglob->gl_pathv[file_گنتی] == عدم) {
            _findclose(لبھو_handle);
            أرجع 2; // Memory allocation for filename failed
        }
        file_گنتی++;
    } طالما (_findnext(لبھو_handle, &لبھو_file_انکڑے) == 0);

    _findclose(لبھو_handle);

    pglob->gl_pathc = file_گنتی;
    أرجع 0;
}

// dirent.h support

#تعریف_کرو MAX_رستہ_لمائی 512
تعريفنوع بنية dirent {
    حرف d_name[MAX_رستہ_لمائی];
} dirent;

تعريفنوع بنية DIR {
    intptr_t handle;
    بنية _finddata_t لبھو_File_انکڑے;
    صحيح پہلا_Read;
} DIR;

ساكن مضمن DIR *opendir(ثابت حرف *name) {
    DIR *dir = (DIR *)حافظہ_دیو(حجم(DIR));
    إذا (dir == عدم) {
        أرجع عدم;
    }

    حرف تلاش_رستہ[MAX_رستہ_لمائی];

    snprintf(تلاش_رستہ, MAX_رستہ_لمائی, "%s\\*.*", name);

    dir->handle = _findfirst(تلاش_رستہ, &dir->لبھو_File_انکڑے);
    إذا (dir->handle == -1) {
        حرر(dir);
        أرجع عدم;
    }

    dir->پہلا_Read = 1;
    أرجع dir;
}

ساكن مضمن بنية dirent *readdir(DIR *directory) {
    ساكن بنية dirent نتیجہ;

    إذا (directory->پہلا_Read) {
        directory->پہلا_Read = 0;
    } وإلا {
        إذا (_findnext(directory->handle, &directory->لبھو_File_انکڑے) != 0) {
            أرجع عدم;
        }
    }

    strncpy(نتیجہ.d_name, directory->لبھو_File_انکڑے.name, MAX_رستہ_لمائی);
    نتیجہ.d_name[MAX_رستہ_لمائی - 1] = '\0'; // Ensure null termination
    أرجع &نتیجہ;
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
#جے_انت // UNISTD_H
