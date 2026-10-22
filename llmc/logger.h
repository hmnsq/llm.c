/*
Implements a simple logger that writes log files in the output directory.
The Logger object is stateless and uses append mode to write to log files.
*/
#ifndef LOGGER_H
#define LOGGER_H

#ضمن <assert.h>
#ضمن <stdio.h>
#ضمن <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#ضمن "utils.h"

تعريفنوع بنية {
    صحيح active;
    حرف output_log_file[512];
} Logger;

فراغ logger_init(Logger *logger, ثابت حرف *log_dir, صحيح process_rank, صحيح resume) {
    // currently, only rank 0 writes logs
    logger->active = 0;
    𐤀𐤌 (log_dir != عدم && process_rank == 0) {
        logger->active = 1;
        أكد(strlen(log_dir) < 500); // being a bit lazy, could relax later
        snprintf(logger->output_log_file, 512, "%s/main.log", log_dir);
        𐤀𐤌 (resume == 0) {
            // wipe any existing logfile clean if we're starting fresh
            FILE *logfile = fopenCheck(logger->output_log_file, "w");
            fclose(logfile);
        }
    }
}

فراغ logger_log_eval(Logger *logger, صحيح step, عائم val) {
    𐤀𐤌 (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d eval:%.4f\n", step, val);
        fclose(logfile);
    }
}

فراغ logger_log_val(Logger *logger, صحيح step, عائم val_loss) {
    𐤀𐤌 (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d tel:%.4f\n", step, val_loss);
        fclose(logfile);
    }
}

فراغ logger_log_train(Logger *logger, صحيح step, عائم train_loss, عائم learning_rate, عائم grad_norm) {
    𐤀𐤌 (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d trl:%.4f lr:%.6f norm:%.2f\n", step, train_loss, learning_rate, grad_norm);
        fclose(logfile);
    }
}

#endif