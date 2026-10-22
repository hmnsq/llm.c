// header file that is necessary to compile on Windows
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ UNISTD_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ UNISTD_H

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ _CRT_SECURE_NO_WARNINGS
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ USE_MATH_ਨਿਰਧਾਰਤ_ਕਰਦਾ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WIN32_LEAN_AND_ਔਸਤ

#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <time.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h> // for malloc and free
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <direct.h> // for _mkdir and _stat
#ਅਨ੍ਤਰ੍ਭਾਵ <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#ਹਦਾਇਤ comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#ਅਨ੍ਤਰ੍ਭਾਵ <winsock2.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CLOCK_MONOTONIC 0
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ clock_gettime(ਪੂਰਨ ignore_variable, ਬਣਤਰ timespec* tv)
{
    ਵਾਪਸ timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OMP /* turn it on */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ F_OK 0
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ access _access

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TURN_OFF_FP_FAST __pragma(float_control( precise, on, ਧੱਕੋ )) // Save current setting and turn on /fp:precise
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TURN_ON_FP_FAST  __pragma(float_control(ਕੱਢੋ)) // Restore file's default settings

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ mkdir(ਰਸਤਾ, ਬਹੁਲਕ) _mkdir(ਰਸਤਾ) /* sketchy way to get mkdir to work on windows */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ stat _stat

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ glob_t {
    size_t gl_pathc;    // Count of matched pathnames
    ਅੱਖਰ **gl_pathv;    // List of matched pathnames
} glob_t;

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ replace_forward_slashes(ਅੱਖਰ* str) {
    ਜਦੋਂ (*str) {
        ਜੇ (*str == '/') {
            *str = '\\';
        }
        str++;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ globfree(glob_t *pglob) {
    ਲਈ (size_t i = 0; i < pglob->gl_pathc; ++i) {
        ਮੁਕਤ(pglob->gl_pathv[i]); // Free the allocated memory for each filename
    }
    ਮੁਕਤ(pglob->gl_pathv); // Free the allocated memory for the list of filenames
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ glob(ਅਚਲ ਅੱਖਰ* ਨਮੂਨਾ, ਪੂਰਨ ignored_flags, ਪੂਰਨ (*ignored_errfunc)(ਅਚਲ ਅੱਖਰ* epath, ਪੂਰਨ eerrno), glob_t* pglob){
    ਬਣਤਰ _finddata_t ਲੱਭੋ_file_ਅੰਕੜੇ;
    ਅੱਖਰ ਭਰਿਆ_ਰਸਤਾ[576]; // stored in pglob->gl_pathv[n]
    ਅੱਖਰ directory_ਰਸਤਾ[512] = {0}; // Store the directory path from the pattern
    ਅੱਖਰ ਨਮੂਨਾ_copy[512]; // Copy of the pattern to modify

    strncpy_s(ਨਮੂਨਾ_copy, ਆਕਾਰ(ਨਮੂਨਾ_copy) - 1, ਨਮੂਨਾ, ਆਕਾਰ(ਨਮੂਨਾ_copy) - 1);

    replace_forward_slashes (ਨਮੂਨਾ_copy); // Replace forward slashes with backslashes

    ਜੇ (strchr(ਨਮੂਨਾ_copy, '\\') != (ਖਾਲੀ*) ਨਿਰਾ) {
        strncpy_s(directory_ਰਸਤਾ, ਆਕਾਰ(directory_ਰਸਤਾ) - 1, ਨਮੂਨਾ_copy, strrchr(ਨਮੂਨਾ_copy, '\\') - ਨਮੂਨਾ_copy + 1);
        directory_ਰਸਤਾ[strrchr(ਨਮੂਨਾ_copy, '\\') - ਨਮੂਨਾ_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t ਲੱਭੋ_handle = _findfirst(ਨਮੂਨਾ_copy, &ਲੱਭੋ_file_ਅੰਕੜੇ);

    ਜੇ (ਲੱਭੋ_handle == -1) {
        ਵਾਪਸ 1; // No files found
    }

    size_t file_ਗਿਣਤੀ = 0;
    size_t max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (ਅੱਖਰ **) ਸਮ੍ਰਿਤੀ_ਦਿਓ(max_files * ਆਕਾਰ(ਅੱਖਰ*)); // freed in globfree

    ਜੇ (pglob->gl_pathv == ਨਿਰਾ) {
        _findclose(ਲੱਭੋ_handle);
        ਵਾਪਸ 2; // Memory allocation failed
    }

    ਕਰੋ {
        ਜੇ (file_ਗਿਣਤੀ >= max_files) {
            _findclose(ਲੱਭੋ_handle);
            ਵਾਪਸ 2; // Too many files found
            }

        snprintf(ਭਰਿਆ_ਰਸਤਾ, ਆਕਾਰ(ਭਰਿਆ_ਰਸਤਾ), "%s%s", directory_ਰਸਤਾ, ਲੱਭੋ_file_ਅੰਕੜੇ.name);

        pglob->gl_pathv[file_ਗਿਣਤੀ] = _strdup(ਭਰਿਆ_ਰਸਤਾ); // freed in globfree

        ਜੇ (pglob->gl_pathv[file_ਗਿਣਤੀ] == ਨਿਰਾ) {
            _findclose(ਲੱਭੋ_handle);
            ਵਾਪਸ 2; // Memory allocation for filename failed
        }
        file_ਗਿਣਤੀ++;
    } ਜਦੋਂ (_findnext(ਲੱਭੋ_handle, &ਲੱਭੋ_file_ਅੰਕੜੇ) == 0);

    _findclose(ਲੱਭੋ_handle);

    pglob->gl_pathc = file_ਗਿਣਤੀ;
    ਵਾਪਸ 0;
}

// dirent.h support

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAX_ਰਸਤਾ_ਲੰਮਾਈ 512
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ dirent {
    ਅੱਖਰ d_name[MAX_ਰਸਤਾ_ਲੰਮਾਈ];
} dirent;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ DIR {
    intptr_t handle;
    ਬਣਤਰ _finddata_t ਲੱਭੋ_File_ਅੰਕੜੇ;
    ਪੂਰਨ ਪਹਿਲਾ_Read;
} DIR;

ਸਥਿਰ ਸਤਰਅੰਦਰ DIR *opendir(ਅਚਲ ਅੱਖਰ *name) {
    DIR *dir = (DIR *)ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਆਕਾਰ(DIR));
    ਜੇ (dir == ਨਿਰਾ) {
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਅੱਖਰ ਖੋਜ_ਰਸਤਾ[MAX_ਰਸਤਾ_ਲੰਮਾਈ];

    snprintf(ਖੋਜ_ਰਸਤਾ, MAX_ਰਸਤਾ_ਲੰਮਾਈ, "%s\\*.*", name);

    dir->handle = _findfirst(ਖੋਜ_ਰਸਤਾ, &dir->ਲੱਭੋ_File_ਅੰਕੜੇ);
    ਜੇ (dir->handle == -1) {
        ਮੁਕਤ(dir);
        ਵਾਪਸ ਨਿਰਾ;
    }

    dir->ਪਹਿਲਾ_Read = 1;
    ਵਾਪਸ dir;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਬਣਤਰ dirent *readdir(DIR *directory) {
    ਸਥਿਰ ਬਣਤਰ dirent ਨਤੀਜਾ;

    ਜੇ (directory->ਪਹਿਲਾ_Read) {
        directory->ਪਹਿਲਾ_Read = 0;
    } ਨਹੀਂਤਾਂ {
        ਜੇ (_findnext(directory->handle, &directory->ਲੱਭੋ_File_ਅੰਕੜੇ) != 0) {
            ਵਾਪਸ ਨਿਰਾ;
        }
    }

    strncpy(ਨਤੀਜਾ.d_name, directory->ਲੱਭੋ_File_ਅੰਕੜੇ.name, MAX_ਰਸਤਾ_ਲੰਮਾਈ);
    ਨਤੀਜਾ.d_name[MAX_ਰਸਤਾ_ਲੰਮਾਈ - 1] = '\0'; // Ensure null termination
    ਵਾਪਸ &ਨਤੀਜਾ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ closedir(DIR *directory) {
    ਜੇ (directory == ਨਿਰਾ) {
        ਵਾਪਸ -1;
    }

    ਜੇ (_findclose(directory->handle) != 0) {
        ਵਾਪਸ -1;
    }

    ਮੁਕਤ(directory);
    ਵਾਪਸ 0;
}
#ਜੇ_ਅੰਤ // UNISTD_H
