/*
Implements a simple logger that writes log files in the output directory.
The Logger object is stateless and uses append mode to write to log files.
*/
#ifndef LOGGER_H
#define LOGGER_H

#شامل <assert.h>
#شامل <stdio.h>
#شامل <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#شامل "utils.h"

تعریفنوع ساختار {
    صحیح active;
    نویسه output_log_file[512];
} Logger;

تهی logger_init(Logger *logger, ثابت نویسه *log_dir, صحیح process_rank, صحیح resume) {
    // currently, only rank 0 writes logs
    logger->active = 0;
    اگر (log_dir != پوچ && process_rank == 0) {
        logger->active = 1;
        أكد(strlen(log_dir) < 500); // being a bit lazy, could relax later
        snprintf(logger->output_log_file, 512, "%s/main.log", log_dir);
        اگر (resume == 0) {
            // wipe any existing logfile clean if we're starting fresh
            FILE *logfile = fopenCheck(logger->output_log_file, "w");
            fclose(logfile);
        }
    }
}

تهی logger_log_eval(Logger *logger, صحیح step, اعشاری val) {
    اگر (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d eval:%.4f\n", step, val);
        fclose(logfile);
    }
}

تهی logger_log_val(Logger *logger, صحیح step, اعشاری val_loss) {
    اگر (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d tel:%.4f\n", step, val_loss);
        fclose(logfile);
    }
}

تهی logger_log_train(Logger *logger, صحیح step, اعشاری train_loss, اعشاری learning_rate, اعشاری grad_norm) {
    اگر (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d trl:%.4f lr:%.6f norm:%.2f\n", step, train_loss, learning_rate, grad_norm);
        fclose(logfile);
    }
}

#endif