// header file that is necessary to compile on Windows
#جيڪڏهن_وصف_نه_ٿيل UNISTD_H
#وصف_ڪريو UNISTD_H

#وصف_ڪريو _CRT_SECURE_NO_WARNINGS
#وصف_ڪريو USE_MATH_مقرر_ڪري
#وصف_ڪريو WIN32_LEAN_AND_سراسري

#ضمن <stdio.h>
#ضمن <math.h>
#ضمن <time.h>
#ضمن <stdlib.h> // for malloc and free
#ضمن <string.h>
#ضمن <direct.h> // for _mkdir and _stat
#ضمن <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#هدايت comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#ضمن <winsock2.h>

#وصف_ڪريو CLOCK_MONOTONIC 0
ساكن مضمن صحيح clock_gettime(صحيح ignore_variable, بنية timespec* tv)
{
    أرجع timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#وصف_ڪريو OMP /* turn it on */
#وصف_ڪريو F_OK 0
#وصف_ڪريو access _access

#وصف_ڪريو TURN_OFF_FP_FAST __pragma(float_control( precise, on, ڌڪو )) // Save current setting and turn on /fp:precise
#وصف_ڪريو TURN_ON_FP_FAST  __pragma(float_control(ڪڍو)) // Restore file's default settings

#وصف_ڪريو mkdir(رستو, گهڻائي) _mkdir(رستو) /* sketchy way to get mkdir to work on windows */
#وصف_ڪريو stat _stat

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

ساكن مضمن صحيح glob(ثابت حرف* نمونو, صحيح ignored_flags, صحيح (*ignored_errfunc)(ثابت حرف* epath, صحيح eerrno), glob_t* pglob){
    بنية _finddata_t ڳوليو_file_معلومات;
    حرف ڀريل_رستو[576]; // stored in pglob->gl_pathv[n]
    حرف directory_رستو[512] = {0}; // Store the directory path from the pattern
    حرف نمونو_copy[512]; // Copy of the pattern to modify

    strncpy_s(نمونو_copy, حجم(نمونو_copy) - 1, نمونو, حجم(نمونو_copy) - 1);

    replace_forward_slashes (نمونو_copy); // Replace forward slashes with backslashes

    إذا (strchr(نمونو_copy, '\\') != (فراغ*) عدم) {
        strncpy_s(directory_رستو, حجم(directory_رستو) - 1, نمونو_copy, strrchr(نمونو_copy, '\\') - نمونو_copy + 1);
        directory_رستو[strrchr(نمونو_copy, '\\') - نمونو_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t ڳوليو_handle = _findfirst(نمونو_copy, &ڳوليو_file_معلومات);

    إذا (ڳوليو_handle == -1) {
        أرجع 1; // No files found
    }

    size_t file_ڳڻپ = 0;
    size_t max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (حرف **) يادگيري_ڏيو(max_files * حجم(حرف*)); // freed in globfree

    إذا (pglob->gl_pathv == عدم) {
        _findclose(ڳوليو_handle);
        أرجع 2; // Memory allocation failed
    }

    افعل {
        إذا (file_ڳڻپ >= max_files) {
            _findclose(ڳوليو_handle);
            أرجع 2; // Too many files found
            }

        snprintf(ڀريل_رستو, حجم(ڀريل_رستو), "%s%s", directory_رستو, ڳوليو_file_معلومات.name);

        pglob->gl_pathv[file_ڳڻپ] = _strdup(ڀريل_رستو); // freed in globfree

        إذا (pglob->gl_pathv[file_ڳڻپ] == عدم) {
            _findclose(ڳوليو_handle);
            أرجع 2; // Memory allocation for filename failed
        }
        file_ڳڻپ++;
    } طالما (_findnext(ڳوليو_handle, &ڳوليو_file_معلومات) == 0);

    _findclose(ڳوليو_handle);

    pglob->gl_pathc = file_ڳڻپ;
    أرجع 0;
}

// dirent.h support

#وصف_ڪريو MAX_رستو_ڊيگهائي 512
تعريفنوع بنية dirent {
    حرف d_name[MAX_رستو_ڊيگهائي];
} dirent;

تعريفنوع بنية DIR {
    intptr_t handle;
    بنية _finddata_t ڳوليو_File_معلومات;
    صحيح پهريون_Read;
} DIR;

ساكن مضمن DIR *opendir(ثابت حرف *name) {
    DIR *dir = (DIR *)يادگيري_ڏيو(حجم(DIR));
    إذا (dir == عدم) {
        أرجع عدم;
    }

    حرف ڳولا_رستو[MAX_رستو_ڊيگهائي];

    snprintf(ڳولا_رستو, MAX_رستو_ڊيگهائي, "%s\\*.*", name);

    dir->handle = _findfirst(ڳولا_رستو, &dir->ڳوليو_File_معلومات);
    إذا (dir->handle == -1) {
        حرر(dir);
        أرجع عدم;
    }

    dir->پهريون_Read = 1;
    أرجع dir;
}

ساكن مضمن بنية dirent *readdir(DIR *directory) {
    ساكن بنية dirent نتيجو;

    إذا (directory->پهريون_Read) {
        directory->پهريون_Read = 0;
    } وإلا {
        إذا (_findnext(directory->handle, &directory->ڳوليو_File_معلومات) != 0) {
            أرجع عدم;
        }
    }

    strncpy(نتيجو.d_name, directory->ڳوليو_File_معلومات.name, MAX_رستو_ڊيگهائي);
    نتيجو.d_name[MAX_رستو_ڊيگهائي - 1] = '\0'; // Ensure null termination
    أرجع &نتيجو;
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
#جيڪڏهن_پورو // UNISTD_H
