/*
Implements a simple logger that writes log files in the output directory.
The Logger object is stateless and uses append mode to write to log files.
*/
#ifndef LOGGER_H
#define LOGGER_H

#അന്തര്ഭാവ <assert.h>
#അന്തര്ഭാവ <stdio.h>
#അന്തര്ഭാവ <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#അന്തര്ഭാവ "utils.h"

തരനിർവചനം ഘടന {
    പൂർണ്ണസംഖ്യ active;
    അക്ഷരം output_log_file[512];
} Logger;

ശൂന്യം logger_init(Logger *logger, സ്ഥിരം അക്ഷരം *log_dir, പൂർണ്ണസംഖ്യ process_rank, പൂർണ്ണസംഖ്യ resume) {
    // currently, only rank 0 writes logs
    logger->active = 0;
    എങ്കിൽ (log_dir != ശൂന്യ && process_rank == 0) {
        logger->active = 1;
        പ്രതിജ്ഞാ(strlen(log_dir) < 500); // being a bit lazy, could relax later
        snprintf(logger->output_log_file, 512, "%s/main.log", log_dir);
        എങ്കിൽ (resume == 0) {
            // wipe any existing logfile clean if we're starting fresh
            FILE *logfile = fopenCheck(logger->output_log_file, "w");
            fclose(logfile);
        }
    }
}

ശൂന്യം logger_log_eval(Logger *logger, പൂർണ്ണസംഖ്യ step, ദശാംശം val) {
    എങ്കിൽ (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d eval:%.4f\n", step, val);
        fclose(logfile);
    }
}

ശൂന്യം logger_log_val(Logger *logger, പൂർണ്ണസംഖ്യ step, ദശാംശം val_loss) {
    എങ്കിൽ (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d tel:%.4f\n", step, val_loss);
        fclose(logfile);
    }
}

ശൂന്യം logger_log_train(Logger *logger, പൂർണ്ണസംഖ്യ step, ദശാംശം train_loss, ദശാംശം learning_rate, ദശാംശം grad_norm) {
    എങ്കിൽ (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d trl:%.4f lr:%.6f norm:%.2f\n", step, train_loss, learning_rate, grad_norm);
        fclose(logfile);
    }
}

#endif