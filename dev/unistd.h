// header file that is necessary to compile on Windows
#अगर_अघोषित UNISTD_H
#घोषणा UNISTD_H

#घोषणा _CRT_SECURE_NO_WARNINGS
#घोषणा _USE_MATH_DEFINES
#घोषणा WIN32_LEAN_AND_MEAN

#समावेश <stdio.h>
#समावेश <math.h>
#समावेश <time.h>
#समावेश <stdlib.h> // for malloc and free
#समावेश <string.h>
#समावेश <direct.h> // for _mkdir and _stat
#समावेश <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#आशय comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#समावेश <winsock2.h>

#घोषणा CLOCK_MONOTONIC 0
अटल अंतरभूत पूर्णांक clock_gettime(पूर्णांक ignore_variable, काष्ठा timespec* tv)
{
    वापस timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#घोषणा OMP /* turn it on */
#घोषणा F_OK 0
#घोषणा access _access

#घोषणा TURN_OFF_FP_FAST __pragma(float_control( precise, on, धकेलो )) // Save current setting and turn on /fp:precise
#घोषणा TURN_ON_FP_FAST  __pragma(float_control(निकालो)) // Restore file's default settings

#घोषणा सूची_गढ़ो(path, mode) _mkdir(path) /* sketchy way to get mkdir to work on windows */
#घोषणा stat _stat

प्रकार काष्ठा glob_t {
    माप_प्रकार gl_pathc;    // Count of matched pathnames
    अक्षर **gl_pathv;    // List of matched pathnames
} glob_t;

अटल अंतरभूत व्योम replace_forward_slashes(अक्षर* str) {
    जबतक (*str) {
        अगर (*str == '/') {
            *str = '\\';
        }
        str++;
    }
}

अटल अंतरभूत व्योम globfree(glob_t *pglob) {
    क्रम (माप_प्रकार i = 0; i < pglob->gl_pathc; ++i) {
        मुक्त(pglob->gl_pathv[i]); // Free the allocated memory for each filename
    }
    मुक्त(pglob->gl_pathv); // Free the allocated memory for the list of filenames
}

अटल अंतरभूत पूर्णांक glob(स्थिर अक्षर* pattern, पूर्णांक ignored_flags, पूर्णांक (*ignored_errfunc)(स्थिर अक्षर* epath, पूर्णांक eerrno), glob_t* pglob){
    काष्ठा _finddata_t find_file_तथ्य;
    अक्षर full_path[576]; // stored in pglob->gl_pathv[n]
    अक्षर directory_path[512] = {0}; // Store the directory path from the pattern
    अक्षर pattern_copy[512]; // Copy of the pattern to modify

    strncpy_s(pattern_copy, माप(pattern_copy) - 1, pattern, माप(pattern_copy) - 1);

    replace_forward_slashes (pattern_copy); // Replace forward slashes with backslashes

    अगर (म_अक्षर(pattern_copy, '\\') != (व्योम*) शून्य) {
        strncpy_s(directory_path, माप(directory_path) - 1, pattern_copy, म_खोजप(pattern_copy, '\\') - pattern_copy + 1);
        directory_path[म_खोजप(pattern_copy, '\\') - pattern_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t find_handle = _findfirst(pattern_copy, &find_file_तथ्य);

    अगर (find_handle == -1) {
        वापस 1; // No files found
    }

    माप_प्रकार file_गिनती = 0;
    माप_प्रकार max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (अक्षर **) दो_स्मृति(max_files * माप(अक्षर*)); // freed in globfree

    अगर (pglob->gl_pathv == शून्य) {
        _findclose(find_handle);
        वापस 2; // Memory allocation failed
    }

    करो {
        अगर (file_गिनती >= max_files) {
            _findclose(find_handle);
            वापस 2; // Too many files found
            }

        snprintf(full_path, माप(full_path), "%s%s", directory_path, find_file_तथ्य.name);

        pglob->gl_pathv[file_गिनती] = _strdup(full_path); // freed in globfree

        अगर (pglob->gl_pathv[file_गिनती] == शून्य) {
            _findclose(find_handle);
            वापस 2; // Memory allocation for filename failed
        }
        file_गिनती++;
    } जबतक (_findnext(find_handle, &find_file_तथ्य) == 0);

    _findclose(find_handle);

    pglob->gl_pathc = file_गिनती;
    वापस 0;
}

// dirent.h support

#घोषणा MAX_PATH_लंबाई 512
प्रकार काष्ठा dirent {
    अक्षर d_name[MAX_PATH_लंबाई];
} dirent;

प्रकार काष्ठा सूची {
    intptr_t handle;
    काष्ठा _finddata_t find_File_तथ्य;
    पूर्णांक firstRead;
} सूची;

अटल अंतरभूत सूची *सूची_खोलो(स्थिर अक्षर *name) {
    सूची *dir = (सूची *)दो_स्मृति(माप(सूची));
    अगर (dir == शून्य) {
        वापस शून्य;
    }

    अक्षर खोज_Path[MAX_PATH_लंबाई];

    snprintf(खोज_Path, MAX_PATH_लंबाई, "%s\\*.*", name);

    dir->handle = _findfirst(खोज_Path, &dir->find_File_तथ्य);
    अगर (dir->handle == -1) {
        मुक्त(dir);
        वापस शून्य;
    }

    dir->firstRead = 1;
    वापस dir;
}

अटल अंतरभूत काष्ठा dirent *सूची_पढ़ो(सूची *directory) {
    अटल काष्ठा dirent परिणाम;

    अगर (directory->firstRead) {
        directory->firstRead = 0;
    } अन्यथा {
        अगर (_findnext(directory->handle, &directory->find_File_तथ्य) != 0) {
            वापस शून्य;
        }
    }

    म_नकलन(परिणाम.d_name, directory->find_File_तथ्य.name, MAX_PATH_लंबाई);
    परिणाम.d_name[MAX_PATH_लंबाई - 1] = '\0'; // Ensure null termination
    वापस &परिणाम;
}

अटल अंतरभूत पूर्णांक बंद_सूची(सूची *directory) {
    अगर (directory == शून्य) {
        वापस -1;
    }

    अगर (_findclose(directory->handle) != 0) {
        वापस -1;
    }

    मुक्त(directory);
    वापस 0;
}
#पूर्ण_अगर // UNISTD_H
