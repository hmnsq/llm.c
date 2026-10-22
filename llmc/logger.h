/*
Implements a simple logger that writes log files in the output directory.
The Logger object is stateless and uses append mode to write to log files.
*/
#ifndef LOGGER_H
#define LOGGER_H

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#ಅನ್ತರ್ಭಾವ "utils.h"

ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ {
    ಪೂರ್ಣಾಙ್ಕ active;
    ವರ್ಣ output_log_file[512];
} Logger;

ಶೂನ್ಯ logger_init(Logger *logger, ಧ್ರುವ ವರ್ಣ *log_dir, ಪೂರ್ಣಾಙ್ಕ process_rank, ಪೂರ್ಣಾಙ್ಕ resume) {
    // currently, only rank 0 writes logs
    logger->active = 0;
    ಯದಿ (log_dir != ರಿಕ್ತ && process_rank == 0) {
        logger->active = 1;
        ಪ್ರತಿಜ್ಞಾ(strlen(log_dir) < 500); // being a bit lazy, could relax later
        snprintf(logger->output_log_file, 512, "%s/main.log", log_dir);
        ಯದಿ (resume == 0) {
            // wipe any existing logfile clean if we're starting fresh
            FILE *logfile = fopenCheck(logger->output_log_file, "w");
            fclose(logfile);
        }
    }
}

ಶೂನ್ಯ logger_log_eval(Logger *logger, ಪೂರ್ಣಾಙ್ಕ step, ಭಿನ್ನ val) {
    ಯದಿ (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d eval:%.4f\n", step, val);
        fclose(logfile);
    }
}

ಶೂನ್ಯ logger_log_val(Logger *logger, ಪೂರ್ಣಾಙ್ಕ step, ಭಿನ್ನ val_loss) {
    ಯದಿ (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d tel:%.4f\n", step, val_loss);
        fclose(logfile);
    }
}

ಶೂನ್ಯ logger_log_train(Logger *logger, ಪೂರ್ಣಾಙ್ಕ step, ಭಿನ್ನ train_loss, ಭಿನ್ನ learning_rate, ಭಿನ್ನ grad_norm) {
    ಯದಿ (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d trl:%.4f lr:%.6f norm:%.2f\n", step, train_loss, learning_rate, grad_norm);
        fclose(logfile);
    }
}

#endif