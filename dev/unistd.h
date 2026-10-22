// header file that is necessary to compile on Windows
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ UNISTD_H
#ವ್ಯಾಖ್ಯಾನಿಸು UNISTD_H

#ವ್ಯಾಖ್ಯಾನಿಸು _CRT_SECURE_NO_WARNINGS
#ವ್ಯಾಖ್ಯಾನಿಸು USE_MATH_ನಿರೂಪಿಸುತ್ತದೆ
#ವ್ಯಾಖ್ಯಾನಿಸು WIN32_LEAN_AND_ಸರಾಸರಿ

#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <time.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h> // for malloc and free
#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <direct.h> // for _mkdir and _stat
#ಅನ್ತರ್ಭಾವ <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#ಸೂಚನೆ comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#ಅನ್ತರ್ಭಾವ <winsock2.h>

#ವ್ಯಾಖ್ಯಾನಿಸು CLOCK_MONOTONIC 0
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ clock_gettime(ಪೂರ್ಣಾಂಕ ignore_variable, ರಚನೆ timespec* tv)
{
    ಹಿಂತಿರುಗಿ timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#ವ್ಯಾಖ್ಯಾನಿಸು OMP /* turn it on */
#ವ್ಯಾಖ್ಯಾನಿಸು F_OK 0
#ವ್ಯಾಖ್ಯಾನಿಸು access _access

#ವ್ಯಾಖ್ಯಾನಿಸು TURN_OFF_FP_FAST __pragma(float_control( precise, on, ತಳ್ಳು )) // Save current setting and turn on /fp:precise
#ವ್ಯಾಖ್ಯಾನಿಸು TURN_ON_FP_FAST  __pragma(float_control(ತೆಗೆ_2)) // Restore file's default settings

#ವ್ಯಾಖ್ಯಾನಿಸು mkdir(ಮಾರ್ಗ, ಬಹುಲಕ) _mkdir(ಮಾರ್ಗ) /* sketchy way to get mkdir to work on windows */
#ವ್ಯಾಖ್ಯಾನಿಸು stat _stat

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ glob_t {
    size_t gl_pathc;    // Count of matched pathnames
    ಅಕ್ಷರ **gl_pathv;    // List of matched pathnames
} glob_t;

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ replace_forward_slashes(ಅಕ್ಷರ* str) {
    ಆಗ (*str) {
        ಆದರೆ (*str == '/') {
            *str = '\\';
        }
        str++;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ globfree(glob_t *pglob) {
    ಗಾಗಿ (size_t i = 0; i < pglob->gl_pathc; ++i) {
        ಬಿಡುಗಡೆ(pglob->gl_pathv[i]); // Free the allocated memory for each filename
    }
    ಬಿಡುಗಡೆ(pglob->gl_pathv); // Free the allocated memory for the list of filenames
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ glob(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* ನಮೂನೆ, ಪೂರ್ಣಾಂಕ ignored_flags, ಪೂರ್ಣಾಂಕ (*ignored_errfunc)(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* epath, ಪೂರ್ಣಾಂಕ eerrno), glob_t* pglob){
    ರಚನೆ _finddata_t ಕಂಡುಹಿಡಿ_file_ದತ್ತಾಂಶ;
    ಅಕ್ಷರ ತುಂಬಿದ_ಮಾರ್ಗ[576]; // stored in pglob->gl_pathv[n]
    ಅಕ್ಷರ directory_ಮಾರ್ಗ[512] = {0}; // Store the directory path from the pattern
    ಅಕ್ಷರ ನಮೂನೆ_copy[512]; // Copy of the pattern to modify

    strncpy_s(ನಮೂನೆ_copy, ಗಾತ್ರ(ನಮೂನೆ_copy) - 1, ನಮೂನೆ, ಗಾತ್ರ(ನಮೂನೆ_copy) - 1);

    replace_forward_slashes (ನಮೂನೆ_copy); // Replace forward slashes with backslashes

    ಆದರೆ (strchr(ನಮೂನೆ_copy, '\\') != (ಶೂನ್ಯ*) ಖಾಲಿ) {
        strncpy_s(directory_ಮಾರ್ಗ, ಗಾತ್ರ(directory_ಮಾರ್ಗ) - 1, ನಮೂನೆ_copy, strrchr(ನಮೂನೆ_copy, '\\') - ನಮೂನೆ_copy + 1);
        directory_ಮಾರ್ಗ[strrchr(ನಮೂನೆ_copy, '\\') - ನಮೂನೆ_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t ಕಂಡುಹಿಡಿ_handle = _findfirst(ನಮೂನೆ_copy, &ಕಂಡುಹಿಡಿ_file_ದತ್ತಾಂಶ);

    ಆದರೆ (ಕಂಡುಹಿಡಿ_handle == -1) {
        ಹಿಂತಿರುಗಿ 1; // No files found
    }

    size_t file_ಎಣಿಕೆ = 0;
    size_t max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (ಅಕ್ಷರ **) ಸ್ಮರಣೆ_ಕೊಡು(max_files * ಗಾತ್ರ(ಅಕ್ಷರ*)); // freed in globfree

    ಆದರೆ (pglob->gl_pathv == ಖಾಲಿ) {
        _findclose(ಕಂಡುಹಿಡಿ_handle);
        ಹಿಂತಿರುಗಿ 2; // Memory allocation failed
    }

    ಮಾಡು {
        ಆದರೆ (file_ಎಣಿಕೆ >= max_files) {
            _findclose(ಕಂಡುಹಿಡಿ_handle);
            ಹಿಂತಿರುಗಿ 2; // Too many files found
            }

        snprintf(ತುಂಬಿದ_ಮಾರ್ಗ, ಗಾತ್ರ(ತುಂಬಿದ_ಮಾರ್ಗ), "%s%s", directory_ಮಾರ್ಗ, ಕಂಡುಹಿಡಿ_file_ದತ್ತಾಂಶ.name);

        pglob->gl_pathv[file_ಎಣಿಕೆ] = _strdup(ತುಂಬಿದ_ಮಾರ್ಗ); // freed in globfree

        ಆದರೆ (pglob->gl_pathv[file_ಎಣಿಕೆ] == ಖಾಲಿ) {
            _findclose(ಕಂಡುಹಿಡಿ_handle);
            ಹಿಂತಿರುಗಿ 2; // Memory allocation for filename failed
        }
        file_ಎಣಿಕೆ++;
    } ಆಗ (_findnext(ಕಂಡುಹಿಡಿ_handle, &ಕಂಡುಹಿಡಿ_file_ದತ್ತಾಂಶ) == 0);

    _findclose(ಕಂಡುಹಿಡಿ_handle);

    pglob->gl_pathc = file_ಎಣಿಕೆ;
    ಹಿಂತಿರುಗಿ 0;
}

// dirent.h support

#ವ್ಯಾಖ್ಯಾನಿಸು MAX_ಮಾರ್ಗ_ಉದ್ದಳತೆ 512
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ dirent {
    ಅಕ್ಷರ d_name[MAX_ಮಾರ್ಗ_ಉದ್ದಳತೆ];
} dirent;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ DIR {
    intptr_t handle;
    ರಚನೆ _finddata_t ಕಂಡುಹಿಡಿ_File_ದತ್ತಾಂಶ;
    ಪೂರ್ಣಾಂಕ ಮೊದಲ_Read;
} DIR;

ಸ್ಥಿರ ಸಾಲೊಳಗೆ DIR *opendir(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *name) {
    DIR *dir = (DIR *)ಸ್ಮರಣೆ_ಕೊಡು(ಗಾತ್ರ(DIR));
    ಆದರೆ (dir == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಅಕ್ಷರ ಹುಡುಕು_ಮಾರ್ಗ[MAX_ಮಾರ್ಗ_ಉದ್ದಳತೆ];

    snprintf(ಹುಡುಕು_ಮಾರ್ಗ, MAX_ಮಾರ್ಗ_ಉದ್ದಳತೆ, "%s\\*.*", name);

    dir->handle = _findfirst(ಹುಡುಕು_ಮಾರ್ಗ, &dir->ಕಂಡುಹಿಡಿ_File_ದತ್ತಾಂಶ);
    ಆದರೆ (dir->handle == -1) {
        ಬಿಡುಗಡೆ(dir);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    dir->ಮೊದಲ_Read = 1;
    ಹಿಂತಿರುಗಿ dir;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ರಚನೆ dirent *readdir(DIR *directory) {
    ಸ್ಥಿರ ರಚನೆ dirent ಫಲಿತಾಂಶ;

    ಆದರೆ (directory->ಮೊದಲ_Read) {
        directory->ಮೊದಲ_Read = 0;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (_findnext(directory->handle, &directory->ಕಂಡುಹಿಡಿ_File_ದತ್ತಾಂಶ) != 0) {
            ಹಿಂತಿರುಗಿ ಖಾಲಿ;
        }
    }

    strncpy(ಫಲಿತಾಂಶ.d_name, directory->ಕಂಡುಹಿಡಿ_File_ದತ್ತಾಂಶ.name, MAX_ಮಾರ್ಗ_ಉದ್ದಳತೆ);
    ಫಲಿತಾಂಶ.d_name[MAX_ಮಾರ್ಗ_ಉದ್ದಳತೆ - 1] = '\0'; // Ensure null termination
    ಹಿಂತಿರುಗಿ &ಫಲಿತಾಂಶ;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ closedir(DIR *directory) {
    ಆದರೆ (directory == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಆದರೆ (_findclose(directory->handle) != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಬಿಡುಗಡೆ(directory);
    ಹಿಂತಿರುಗಿ 0;
}
#ಆದರೆ_ಅಂತ್ಯ // UNISTD_H
