/*
Implements a simple logger that writes log files in the output directory.
The Logger object is stateless and uses append mode to write to log files.
*/
#ifndef LOGGER_H
#define LOGGER_H

#அந்தர்பாவ <assert.h>
#அந்தர்பாவ <stdio.h>
#அந்தர்பாவ <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#அந்தர்பாவ "utils.h"

வகைவரை கட்டமைப்பு {
    முழு active;
    எழுத்து output_log_file[512];
} Logger;

வெறுமை logger_init(Logger *logger, மாறிலி எழுத்து *log_dir, முழு process_rank, முழு resume) {
    // currently, only rank 0 writes logs
    logger->active = 0;
    எனில் (log_dir != வெற்று && process_rank == 0) {
        logger->active = 1;
        ப்ரதிஜ்ஞா(strlen(log_dir) < 500); // being a bit lazy, could relax later
        snprintf(logger->output_log_file, 512, "%s/main.log", log_dir);
        எனில் (resume == 0) {
            // wipe any existing logfile clean if we're starting fresh
            FILE *logfile = fopenCheck(logger->output_log_file, "w");
            fclose(logfile);
        }
    }
}

வெறுமை logger_log_eval(Logger *logger, முழு step, தசம val) {
    எனில் (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d eval:%.4f\n", step, val);
        fclose(logfile);
    }
}

வெறுமை logger_log_val(Logger *logger, முழு step, தசம val_loss) {
    எனில் (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d tel:%.4f\n", step, val_loss);
        fclose(logfile);
    }
}

வெறுமை logger_log_train(Logger *logger, முழு step, தசம train_loss, தசம learning_rate, தசம grad_norm) {
    எனில் (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d trl:%.4f lr:%.6f norm:%.2f\n", step, train_loss, learning_rate, grad_norm);
        fclose(logfile);
    }
}

#endif