// header file that is necessary to compile on Windows
#ifndef UNISTD_H
#define UNISTD_H

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define WIN32_LEAN_AND_MEAN

#અન્તર્ભાવ <stdio.h>
#અન્તર્ભાવ <math.h>
#અન્તર્ભાવ <time.h>
#અન્તર્ભાવ <stdlib.h> // for malloc and free
#અન્તર્ભાવ <string.h>
#અન્તર્ભાવ <direct.h> // for _mkdir and _stat
#અન્તર્ભાવ <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#pragma comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#અન્તર્ભાવ <winsock2.h>

#define CLOCK_MONOTONIC 0
સ્થિર અંતઃરેખ પૂર્ણાંક clock_gettime(પૂર્ણાંક ignore_variable, સંરચના timespec* tv)
{
    પાછા timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#define OMP /* turn it on */
#define F_OK 0
#define access _access

#define TURN_OFF_FP_FAST __pragma(float_control( precise, on, ધકેલો )) // Save current setting and turn on /fp:precise
#define TURN_ON_FP_FAST  __pragma(float_control(કાઢો)) // Restore file's default settings

#define mkdir(path, mode) _mkdir(path) /* sketchy way to get mkdir to work on windows */
#define stat _stat

પ્રકારવ્યાખ્યા સંરચના glob_t {
    size_t gl_pathc;    // Count of matched pathnames
    અક્ષર **gl_pathv;    // List of matched pathnames
} glob_t;

સ્થિર અંતઃરેખ શૂન્ય replace_forward_slashes(અક્ષર* str) {
    જ્યારે (*str) {
        જો (*str == '/') {
            *str = '\\';
        }
        str++;
    }
}

સ્થિર અંતઃરેખ શૂન્ય globfree(glob_t *pglob) {
    માટે (size_t i = 0; i < pglob->gl_pathc; ++i) {
        મુક્ત(pglob->gl_pathv[i]); // Free the allocated memory for each filename
    }
    મુક્ત(pglob->gl_pathv); // Free the allocated memory for the list of filenames
}

સ્થિર અંતઃરેખ પૂર્ણાંક glob(અચલ અક્ષર* pattern, પૂર્ણાંક ignored_flags, પૂર્ણાંક (*ignored_errfunc)(અચલ અક્ષર* epath, પૂર્ણાંક eerrno), glob_t* pglob){
    સંરચના _finddata_t find_file_data;
    અક્ષર full_path[576]; // stored in pglob->gl_pathv[n]
    અક્ષર directory_path[512] = {0}; // Store the directory path from the pattern
    અક્ષર pattern_copy[512]; // Copy of the pattern to modify

    strncpy_s(pattern_copy, કદ(pattern_copy) - 1, pattern, કદ(pattern_copy) - 1);

    replace_forward_slashes (pattern_copy); // Replace forward slashes with backslashes

    જો (strchr(pattern_copy, '\\') != (શૂન્ય*) ખાલી) {
        strncpy_s(directory_path, કદ(directory_path) - 1, pattern_copy, strrchr(pattern_copy, '\\') - pattern_copy + 1);
        directory_path[strrchr(pattern_copy, '\\') - pattern_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t find_handle = _findfirst(pattern_copy, &find_file_data);

    જો (find_handle == -1) {
        પાછા 1; // No files found
    }

    size_t file_ગણતરી = 0;
    size_t max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (અક્ષર **) malloc(max_files * કદ(અક્ષર*)); // freed in globfree

    જો (pglob->gl_pathv == ખાલી) {
        _findclose(find_handle);
        પાછા 2; // Memory allocation failed
    }

    કરો {
        જો (file_ગણતરી >= max_files) {
            _findclose(find_handle);
            પાછા 2; // Too many files found
            }

        snprintf(full_path, કદ(full_path), "%s%s", directory_path, find_file_data.name);

        pglob->gl_pathv[file_ગણતરી] = _strdup(full_path); // freed in globfree

        જો (pglob->gl_pathv[file_ગણતરી] == ખાલી) {
            _findclose(find_handle);
            પાછા 2; // Memory allocation for filename failed
        }
        file_ગણતરી++;
    } જ્યારે (_findnext(find_handle, &find_file_data) == 0);

    _findclose(find_handle);

    pglob->gl_pathc = file_ગણતરી;
    પાછા 0;
}

// dirent.h support

#define MAX_PATH_LENGTH 512
પ્રકારવ્યાખ્યા સંરચના dirent {
    અક્ષર d_name[MAX_PATH_LENGTH];
} dirent;

પ્રકારવ્યાખ્યા સંરચના DIR {
    intptr_t handle;
    સંરચના _finddata_t findFileData;
    પૂર્ણાંક firstRead;
} DIR;

સ્થિર અંતઃરેખ DIR *opendir(અચલ અક્ષર *name) {
    DIR *dir = (DIR *)malloc(કદ(DIR));
    જો (dir == ખાલી) {
        પાછા ખાલી;
    }

    અક્ષર શોધ_Path[MAX_PATH_LENGTH];

    snprintf(શોધ_Path, MAX_PATH_LENGTH, "%s\\*.*", name);

    dir->handle = _findfirst(શોધ_Path, &dir->findFileData);
    જો (dir->handle == -1) {
        મુક્ત(dir);
        પાછા ખાલી;
    }

    dir->firstRead = 1;
    પાછા dir;
}

સ્થિર અંતઃરેખ સંરચના dirent *readdir(DIR *directory) {
    સ્થિર સંરચના dirent result;

    જો (directory->firstRead) {
        directory->firstRead = 0;
    } નહીંતર {
        જો (_findnext(directory->handle, &directory->findFileData) != 0) {
            પાછા ખાલી;
        }
    }

    strncpy(result.d_name, directory->findFileData.name, MAX_PATH_LENGTH);
    result.d_name[MAX_PATH_LENGTH - 1] = '\0'; // Ensure null termination
    પાછા &result;
}

સ્થિર અંતઃરેખ પૂર્ણાંક closedir(DIR *directory) {
    જો (directory == ખાલી) {
        પાછા -1;
    }

    જો (_findclose(directory->handle) != 0) {
        પાછા -1;
    }

    મુક્ત(directory);
    પાછા 0;
}
#endif // UNISTD_H
