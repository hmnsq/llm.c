/*
Implements a simple logger that writes log files in the output directory.
The Logger object is stateless and uses append mode to write to log files.
*/
#ifndef LOGGER_H
#define LOGGER_H

#include <assert.h>
#include <stdio.h>
#include <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "utils.h"

typedef sa_nracanaa {
    ຈຳນວນເຕັມ active;
    var_na output_log_file[512];
} Logger;

void logger_init(Logger *logger, const var_na *log_dir, ຈຳນວນເຕັມ process_rank, ຈຳນວນເຕັມ resume) {
    // currently, only rank 0 writes logs
    logger->active = 0;
    ຖ້າ (log_dir != NULL && process_rank == 0) {
        logger->active = 1;
        assert(strlen(log_dir) < 500); // being a bit lazy, could relax later
        snprintf(logger->output_log_file, 512, "%s/main.log", log_dir);
        ຖ້າ (resume == 0) {
            // wipe any existing logfile clean if we're starting fresh
            FILE *logfile = fopenCheck(logger->output_log_file, "w");
            fclose(logfile);
        }
    }
}

void logger_log_eval(Logger *logger, ຈຳນວນເຕັມ step, float val) {
    ຖ້າ (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d eval:%.4f\n", step, val);
        fclose(logfile);
    }
}

void logger_log_val(Logger *logger, ຈຳນວນເຕັມ step, float val_loss) {
    ຖ້າ (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d tel:%.4f\n", step, val_loss);
        fclose(logfile);
    }
}

void logger_log_train(Logger *logger, ຈຳນວນເຕັມ step, float train_loss, float learning_rate, float grad_norm) {
    ຖ້າ (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d trl:%.4f lr:%.6f norm:%.2f\n", step, train_loss, learning_rate, grad_norm);
        fclose(logfile);
    }
}

#endif