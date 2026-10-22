// header file that is necessary to compile on Windows
#ifndef UNISTD_H
#define UNISTD_H

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define WIN32_LEAN_AND_MEAN

#अन्तर्भाव <stdio.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव <time.h>
#अन्तर्भाव <stdlib.h> // for malloc and free
#अन्तर्भाव <string.h>
#अन्तर्भाव <direct.h> // for _mkdir and _stat
#अन्तर्भाव <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#pragma comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#अन्तर्भाव <winsock2.h>

#define CLOCK_MONOTONIC 0
स्थिर अन्तःस्थ पूर्णाङ्क clock_gettime(पूर्णाङ्क ignore_variable, संरचना timespec* tv)
{
    फर्काऊ timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#define OMP /* turn it on */
#define F_OK 0
#define access _access

#define TURN_OFF_FP_FAST __pragma(float_control( precise, on, थप )) // Save current setting and turn on /fp:precise
#define TURN_ON_FP_FAST  __pragma(float_control(झिक)) // Restore file's default settings

#define सूची_गढ़ो(path, mode) _mkdir(path) /* sketchy way to get mkdir to work on windows */
#define stat _stat

प्रकारनिर्देश संरचना glob_t {
    माप_प्रकार gl_pathc;    // Count of matched pathnames
    वर्ण **gl_pathv;    // List of matched pathnames
} glob_t;

स्थिर अन्तःस्थ शून्य replace_forward_slashes(वर्ण* str) {
    जबसम्म (*str) {
        यदि (*str == '/') {
            *str = '\\';
        }
        str++;
    }
}

स्थिर अन्तःस्थ शून्य globfree(glob_t *pglob) {
    लागि (माप_प्रकार i = 0; i < pglob->gl_pathc; ++i) {
        मुक्त(pglob->gl_pathv[i]); // Free the allocated memory for each filename
    }
    मुक्त(pglob->gl_pathv); // Free the allocated memory for the list of filenames
}

स्थिर अन्तःस्थ पूर्णाङ्क glob(अचर वर्ण* pattern, पूर्णाङ्क ignored_flags, पूर्णाङ्क (*ignored_errfunc)(अचर वर्ण* epath, पूर्णाङ्क eerrno), glob_t* pglob){
    संरचना _finddata_t find_file_data;
    वर्ण full_path[576]; // stored in pglob->gl_pathv[n]
    वर्ण directory_path[512] = {0}; // Store the directory path from the pattern
    वर्ण pattern_copy[512]; // Copy of the pattern to modify

    strncpy_s(pattern_copy, परिमाण(pattern_copy) - 1, pattern, परिमाण(pattern_copy) - 1);

    replace_forward_slashes (pattern_copy); // Replace forward slashes with backslashes

    यदि (म_अक्षर(pattern_copy, '\\') != (शून्य*) रिक्त) {
        strncpy_s(directory_path, परिमाण(directory_path) - 1, pattern_copy, म_खोजप(pattern_copy, '\\') - pattern_copy + 1);
        directory_path[म_खोजप(pattern_copy, '\\') - pattern_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t find_handle = _findfirst(pattern_copy, &find_file_data);

    यदि (find_handle == -1) {
        फर्काऊ 1; // No files found
    }

    माप_प्रकार file_गन्ती = 0;
    माप_प्रकार max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (वर्ण **) दो_स्मृति(max_files * परिमाण(वर्ण*)); // freed in globfree

    यदि (pglob->gl_pathv == रिक्त) {
        _findclose(find_handle);
        फर्काऊ 2; // Memory allocation failed
    }

    गर {
        यदि (file_गन्ती >= max_files) {
            _findclose(find_handle);
            फर्काऊ 2; // Too many files found
            }

        snprintf(full_path, परिमाण(full_path), "%s%s", directory_path, find_file_data.name);

        pglob->gl_pathv[file_गन्ती] = _strdup(full_path); // freed in globfree

        यदि (pglob->gl_pathv[file_गन्ती] == रिक्त) {
            _findclose(find_handle);
            फर्काऊ 2; // Memory allocation for filename failed
        }
        file_गन्ती++;
    } जबसम्म (_findnext(find_handle, &find_file_data) == 0);

    _findclose(find_handle);

    pglob->gl_pathc = file_गन्ती;
    फर्काऊ 0;
}

// dirent.h support

#define MAX_PATH_LENGTH 512
प्रकारनिर्देश संरचना dirent {
    वर्ण d_name[MAX_PATH_LENGTH];
} dirent;

प्रकारनिर्देश संरचना सूची {
    intptr_t handle;
    संरचना _finddata_t findFileData;
    पूर्णाङ्क firstRead;
} सूची;

स्थिर अन्तःस्थ सूची *सूची_खोलो(अचर वर्ण *name) {
    सूची *dir = (सूची *)दो_स्मृति(परिमाण(सूची));
    यदि (dir == रिक्त) {
        फर्काऊ रिक्त;
    }

    वर्ण खोज_Path[MAX_PATH_LENGTH];

    snprintf(खोज_Path, MAX_PATH_LENGTH, "%s\\*.*", name);

    dir->handle = _findfirst(खोज_Path, &dir->findFileData);
    यदि (dir->handle == -1) {
        मुक्त(dir);
        फर्काऊ रिक्त;
    }

    dir->firstRead = 1;
    फर्काऊ dir;
}

स्थिर अन्तःस्थ संरचना dirent *सूची_पढ़ो(सूची *directory) {
    स्थिर संरचना dirent result;

    यदि (directory->firstRead) {
        directory->firstRead = 0;
    } अन्यथा {
        यदि (_findnext(directory->handle, &directory->findFileData) != 0) {
            फर्काऊ रिक्त;
        }
    }

    म_नकलन(result.d_name, directory->findFileData.name, MAX_PATH_LENGTH);
    result.d_name[MAX_PATH_LENGTH - 1] = '\0'; // Ensure null termination
    फर्काऊ &result;
}

स्थिर अन्तःस्थ पूर्णाङ्क बंद_सूची(सूची *directory) {
    यदि (directory == रिक्त) {
        फर्काऊ -1;
    }

    यदि (_findclose(directory->handle) != 0) {
        फर्काऊ -1;
    }

    मुक्त(directory);
    फर्काऊ 0;
}
#endif // UNISTD_H
