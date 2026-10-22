// header file that is necessary to compile on Windows
#ifndef UNISTD_H
#define UNISTD_H

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define WIN32_LEAN_AND_MEAN

#ଅନ୍ତର୍ଭାଵ <stdio.h>
#ଅନ୍ତର୍ଭାଵ <math.h>
#ଅନ୍ତର୍ଭାଵ <time.h>
#ଅନ୍ତର୍ଭାଵ <stdlib.h> // for malloc and free
#ଅନ୍ତର୍ଭାଵ <string.h>
#ଅନ୍ତର୍ଭାଵ <direct.h> // for _mkdir and _stat
#ଅନ୍ତର୍ଭାଵ <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#pragma comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#ଅନ୍ତର୍ଭାଵ <winsock2.h>

#define CLOCK_MONOTONIC 0
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା clock_gettime(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ignore_variable, ସଂରଚନା timespec* tv)
{
    ଫେରନ୍ତୁ timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#define OMP /* turn it on */
#define F_OK 0
#define access _access

#define TURN_OFF_FP_FAST __pragma(float_control( precise, on, ଠେଲ )) // Save current setting and turn on /fp:precise
#define TURN_ON_FP_FAST  __pragma(float_control(କାଢ଼)) // Restore file's default settings

#define mkdir(path, mode) _mkdir(path) /* sketchy way to get mkdir to work on windows */
#define stat _stat

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା glob_t {
    size_t gl_pathc;    // Count of matched pathnames
    ଅକ୍ଷର **gl_pathv;    // List of matched pathnames
} glob_t;

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ replace_forward_slashes(ଅକ୍ଷର* str) {
    ଯେତେବେଳେ (*str) {
        ଯଦି (*str == '/') {
            *str = '\\';
        }
        str++;
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ globfree(glob_t *pglob) {
    ପାଇଁ (size_t i = 0; i < pglob->gl_pathc; ++i) {
        ମୁକ୍ତ(pglob->gl_pathv[i]); // Free the allocated memory for each filename
    }
    ମୁକ୍ତ(pglob->gl_pathv); // Free the allocated memory for the list of filenames
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା glob(ଅଚଳ ଅକ୍ଷର* pattern, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ignored_flags, ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*ignored_errfunc)(ଅଚଳ ଅକ୍ଷର* epath, ପୂର୍ଣ୍ଣସଂଖ୍ୟା eerrno), glob_t* pglob){
    ସଂରଚନା _finddata_t find_file_data;
    ଅକ୍ଷର full_path[576]; // stored in pglob->gl_pathv[n]
    ଅକ୍ଷର directory_path[512] = {0}; // Store the directory path from the pattern
    ଅକ୍ଷର pattern_copy[512]; // Copy of the pattern to modify

    strncpy_s(pattern_copy, ଆକାର(pattern_copy) - 1, pattern, ଆକାର(pattern_copy) - 1);

    replace_forward_slashes (pattern_copy); // Replace forward slashes with backslashes

    ଯଦି (strchr(pattern_copy, '\\') != (ଶୂନ୍ୟ*) ଖାଲି) {
        strncpy_s(directory_path, ଆକାର(directory_path) - 1, pattern_copy, strrchr(pattern_copy, '\\') - pattern_copy + 1);
        directory_path[strrchr(pattern_copy, '\\') - pattern_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t find_handle = _findfirst(pattern_copy, &find_file_data);

    ଯଦି (find_handle == -1) {
        ଫେରନ୍ତୁ 1; // No files found
    }

    size_t file_ଗଣନା = 0;
    size_t max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (ଅକ୍ଷର **) malloc(max_files * ଆକାର(ଅକ୍ଷର*)); // freed in globfree

    ଯଦି (pglob->gl_pathv == ଖାଲି) {
        _findclose(find_handle);
        ଫେରନ୍ତୁ 2; // Memory allocation failed
    }

    କର {
        ଯଦି (file_ଗଣନା >= max_files) {
            _findclose(find_handle);
            ଫେରନ୍ତୁ 2; // Too many files found
            }

        snprintf(full_path, ଆକାର(full_path), "%s%s", directory_path, find_file_data.name);

        pglob->gl_pathv[file_ଗଣନା] = _strdup(full_path); // freed in globfree

        ଯଦି (pglob->gl_pathv[file_ଗଣନା] == ଖାଲି) {
            _findclose(find_handle);
            ଫେରନ୍ତୁ 2; // Memory allocation for filename failed
        }
        file_ଗଣନା++;
    } ଯେତେବେଳେ (_findnext(find_handle, &find_file_data) == 0);

    _findclose(find_handle);

    pglob->gl_pathc = file_ଗଣନା;
    ଫେରନ୍ତୁ 0;
}

// dirent.h support

#define MAX_PATH_LENGTH 512
ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା dirent {
    ଅକ୍ଷର d_name[MAX_PATH_LENGTH];
} dirent;

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା DIR {
    intptr_t handle;
    ସଂରଚନା _finddata_t findFileData;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା firstRead;
} DIR;

ସ୍ଥିର ଧାଡ଼ିଭିତରେ DIR *opendir(ଅଚଳ ଅକ୍ଷର *name) {
    DIR *dir = (DIR *)malloc(ଆକାର(DIR));
    ଯଦି (dir == ଖାଲି) {
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ଅକ୍ଷର ସନ୍ଧାନ_Path[MAX_PATH_LENGTH];

    snprintf(ସନ୍ଧାନ_Path, MAX_PATH_LENGTH, "%s\\*.*", name);

    dir->handle = _findfirst(ସନ୍ଧାନ_Path, &dir->findFileData);
    ଯଦି (dir->handle == -1) {
        ମୁକ୍ତ(dir);
        ଫେରନ୍ତୁ ଖାଲି;
    }

    dir->firstRead = 1;
    ଫେରନ୍ତୁ dir;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ସଂରଚନା dirent *readdir(DIR *directory) {
    ସ୍ଥିର ସଂରଚନା dirent result;

    ଯଦି (directory->firstRead) {
        directory->firstRead = 0;
    } ଅନ୍ୟଥା {
        ଯଦି (_findnext(directory->handle, &directory->findFileData) != 0) {
            ଫେରନ୍ତୁ ଖାଲି;
        }
    }

    strncpy(result.d_name, directory->findFileData.name, MAX_PATH_LENGTH);
    result.d_name[MAX_PATH_LENGTH - 1] = '\0'; // Ensure null termination
    ଫେରନ୍ତୁ &result;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା closedir(DIR *directory) {
    ଯଦି (directory == ଖାଲି) {
        ଫେରନ୍ତୁ -1;
    }

    ଯଦି (_findclose(directory->handle) != 0) {
        ଫେରନ୍ତୁ -1;
    }

    ମୁକ୍ତ(directory);
    ଫେରନ୍ତୁ 0;
}
#endif // UNISTD_H
