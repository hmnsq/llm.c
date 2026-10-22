// header file that is necessary to compile on Windows
#నిర్వచితం_కాకపోతే UNISTD_H
#నిర్వచించు UNISTD_H

#నిర్వచించు _CRT_SECURE_NO_WARNINGS
#నిర్వచించు USE_MATH_నిర్వచిస్తుంది
#నిర్వచించు WIN32_LEAN_AND_సగటు

#అన్తర్భావ <stdio.h>
#అన్తర్భావ <math.h>
#అన్తర్భావ <time.h>
#అన్తర్భావ <stdlib.h> // for malloc and free
#అన్తర్భావ <string.h>
#అన్తర్భావ <direct.h> // for _mkdir and _stat
#అన్తర్భావ <io.h> // needed for _access below and _findfirst, _findnext, _findclose
#సూచన comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#అన్తర్భావ <winsock2.h>

#నిర్వచించు CLOCK_MONOTONIC 0
స్థిర అంతఃవరుస పూర్ణాంకం clock_gettime(పూర్ణాంకం ignore_variable, నిర్మాణం timespec* tv)
{
    తిరిగి timespec_get(tv, TIME_UTC); // TODO: not sure this is the best solution. Need to review.
}

#నిర్వచించు OMP /* turn it on */
#నిర్వచించు F_OK 0
#నిర్వచించు access _access

#నిర్వచించు TURN_OFF_FP_FAST __pragma(float_control( precise, on, నెట్టు )) // Save current setting and turn on /fp:precise
#నిర్వచించు TURN_ON_FP_FAST  __pragma(float_control(తీయి)) // Restore file's default settings

#నిర్వచించు mkdir(మార్గం, బహుళకం) _mkdir(మార్గం) /* sketchy way to get mkdir to work on windows */
#నిర్వచించు stat _stat

రకనిర్వచనం నిర్మాణం glob_t {
    size_t gl_pathc;    // Count of matched pathnames
    అక్షరం **gl_pathv;    // List of matched pathnames
} glob_t;

స్థిర అంతఃవరుస శూన్య replace_forward_slashes(అక్షరం* str) {
    అయితే (*str) {
        ఒకవేళ (*str == '/') {
            *str = '\\';
        }
        str++;
    }
}

స్థిర అంతఃవరుస శూన్య globfree(glob_t *pglob) {
    కొరకు (size_t i = 0; i < pglob->gl_pathc; ++i) {
        విడుదల(pglob->gl_pathv[i]); // Free the allocated memory for each filename
    }
    విడుదల(pglob->gl_pathv); // Free the allocated memory for the list of filenames
}

స్థిర అంతఃవరుస పూర్ణాంకం glob(స్థిరరాశి అక్షరం* నమూనా, పూర్ణాంకం ignored_flags, పూర్ణాంకం (*ignored_errfunc)(స్థిరరాశి అక్షరం* epath, పూర్ణాంకం eerrno), glob_t* pglob){
    నిర్మాణం _finddata_t కనుగొను_file_దత్తాంశం;
    అక్షరం నిండిన_మార్గం[576]; // stored in pglob->gl_pathv[n]
    అక్షరం directory_మార్గం[512] = {0}; // Store the directory path from the pattern
    అక్షరం నమూనా_copy[512]; // Copy of the pattern to modify

    strncpy_s(నమూనా_copy, పరిమాణం(నమూనా_copy) - 1, నమూనా, పరిమాణం(నమూనా_copy) - 1);

    replace_forward_slashes (నమూనా_copy); // Replace forward slashes with backslashes

    ఒకవేళ (strchr(నమూనా_copy, '\\') != (శూన్య*) ఖాళీ) {
        strncpy_s(directory_మార్గం, పరిమాణం(directory_మార్గం) - 1, నమూనా_copy, strrchr(నమూనా_copy, '\\') - నమూనా_copy + 1);
        directory_మార్గం[strrchr(నమూనా_copy, '\\') - నమూనా_copy + 1] = '\0';
    }

    // find the first file matching the pattern in the directory
    intptr_t కనుగొను_handle = _findfirst(నమూనా_copy, &కనుగొను_file_దత్తాంశం);

    ఒకవేళ (కనుగొను_handle == -1) {
        తిరిగి 1; // No files found
    }

    size_t file_లెక్క = 0;
    size_t max_files = 64000; // hard-coded limit for the number of files

    pglob->gl_pathv = (అక్షరం **) జ్ఞాపకం_ఇవ్వు(max_files * పరిమాణం(అక్షరం*)); // freed in globfree

    ఒకవేళ (pglob->gl_pathv == ఖాళీ) {
        _findclose(కనుగొను_handle);
        తిరిగి 2; // Memory allocation failed
    }

    చేయి {
        ఒకవేళ (file_లెక్క >= max_files) {
            _findclose(కనుగొను_handle);
            తిరిగి 2; // Too many files found
            }

        snprintf(నిండిన_మార్గం, పరిమాణం(నిండిన_మార్గం), "%s%s", directory_మార్గం, కనుగొను_file_దత్తాంశం.name);

        pglob->gl_pathv[file_లెక్క] = _strdup(నిండిన_మార్గం); // freed in globfree

        ఒకవేళ (pglob->gl_pathv[file_లెక్క] == ఖాళీ) {
            _findclose(కనుగొను_handle);
            తిరిగి 2; // Memory allocation for filename failed
        }
        file_లెక్క++;
    } అయితే (_findnext(కనుగొను_handle, &కనుగొను_file_దత్తాంశం) == 0);

    _findclose(కనుగొను_handle);

    pglob->gl_pathc = file_లెక్క;
    తిరిగి 0;
}

// dirent.h support

#నిర్వచించు MAX_మార్గం_నిడివి 512
రకనిర్వచనం నిర్మాణం dirent {
    అక్షరం d_name[MAX_మార్గం_నిడివి];
} dirent;

రకనిర్వచనం నిర్మాణం DIR {
    intptr_t handle;
    నిర్మాణం _finddata_t కనుగొను_File_దత్తాంశం;
    పూర్ణాంకం మొదటి_Read;
} DIR;

స్థిర అంతఃవరుస DIR *opendir(స్థిరరాశి అక్షరం *name) {
    DIR *dir = (DIR *)జ్ఞాపకం_ఇవ్వు(పరిమాణం(DIR));
    ఒకవేళ (dir == ఖాళీ) {
        తిరిగి ఖాళీ;
    }

    అక్షరం శోధన_మార్గం[MAX_మార్గం_నిడివి];

    snprintf(శోధన_మార్గం, MAX_మార్గం_నిడివి, "%s\\*.*", name);

    dir->handle = _findfirst(శోధన_మార్గం, &dir->కనుగొను_File_దత్తాంశం);
    ఒకవేళ (dir->handle == -1) {
        విడుదల(dir);
        తిరిగి ఖాళీ;
    }

    dir->మొదటి_Read = 1;
    తిరిగి dir;
}

స్థిర అంతఃవరుస నిర్మాణం dirent *readdir(DIR *directory) {
    స్థిర నిర్మాణం dirent ఫలితం;

    ఒకవేళ (directory->మొదటి_Read) {
        directory->మొదటి_Read = 0;
    } లేకపోతే {
        ఒకవేళ (_findnext(directory->handle, &directory->కనుగొను_File_దత్తాంశం) != 0) {
            తిరిగి ఖాళీ;
        }
    }

    strncpy(ఫలితం.d_name, directory->కనుగొను_File_దత్తాంశం.name, MAX_మార్గం_నిడివి);
    ఫలితం.d_name[MAX_మార్గం_నిడివి - 1] = '\0'; // Ensure null termination
    తిరిగి &ఫలితం;
}

స్థిర అంతఃవరుస పూర్ణాంకం closedir(DIR *directory) {
    ఒకవేళ (directory == ఖాళీ) {
        తిరిగి -1;
    }

    ఒకవేళ (_findclose(directory->handle) != 0) {
        తిరిగి -1;
    }

    విడుదల(directory);
    తిరిగి 0;
}
#అయితే_ముగింపు // UNISTD_H
