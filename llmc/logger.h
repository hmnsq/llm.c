/*
Implements a simple logger that writes log files in the output directory.
The Logger object is stateless and uses append mode to write to log files.
*/
#ifndef LOGGER_H
#define LOGGER_H

#ଅନ୍ତର୍ଭାଵ <assert.h>
#ଅନ୍ତର୍ଭାଵ <stdio.h>
#ଅନ୍ତର୍ଭାଵ <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#ଅନ୍ତର୍ଭାଵ "utils.h"

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା active;
    ଅକ୍ଷର output_log_file[512];
} Logger;

ଶୂନ୍ୟ logger_init(Logger *logger, ଅଚଳ ଅକ୍ଷର *log_dir, ପୂର୍ଣ୍ଣସଂଖ୍ୟା process_rank, ପୂର୍ଣ୍ଣସଂଖ୍ୟା resume) {
    // currently, only rank 0 writes logs
    logger->active = 0;
    ଯଦି (log_dir != ଖାଲି && process_rank == 0) {
        logger->active = 1;
        ପ୍ରତିଜ୍ଞା(strlen(log_dir) < 500); // being a bit lazy, could relax later
        snprintf(logger->output_log_file, 512, "%s/main.log", log_dir);
        ଯଦି (resume == 0) {
            // wipe any existing logfile clean if we're starting fresh
            FILE *logfile = fopenCheck(logger->output_log_file, "w");
            fclose(logfile);
        }
    }
}

ଶୂନ୍ୟ logger_log_eval(Logger *logger, ପୂର୍ଣ୍ଣସଂଖ୍ୟା step, ଦଶମିକ val) {
    ଯଦି (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d eval:%.4f\n", step, val);
        fclose(logfile);
    }
}

ଶୂନ୍ୟ logger_log_val(Logger *logger, ପୂର୍ଣ୍ଣସଂଖ୍ୟା step, ଦଶମିକ val_loss) {
    ଯଦି (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d tel:%.4f\n", step, val_loss);
        fclose(logfile);
    }
}

ଶୂନ୍ୟ logger_log_train(Logger *logger, ପୂର୍ଣ୍ଣସଂଖ୍ୟା step, ଦଶମିକ train_loss, ଦଶମିକ learning_rate, ଦଶମିକ grad_norm) {
    ଯଦି (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d trl:%.4f lr:%.6f norm:%.2f\n", step, train_loss, learning_rate, grad_norm);
        fclose(logfile);
    }
}

#endif