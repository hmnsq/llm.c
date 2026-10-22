/*
Implements a simple logger that writes log files in the output directory.
The Logger object is stateless and uses append mode to write to log files.
*/
#ifndef LOGGER_H
#define LOGGER_H

#অন্তর্ভাব <assert.h>
#অন্তর্ভাব <stdio.h>
#অন্তর্ভাব <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#অন্তর্ভাব "utils.h"

প্রকারনির্দেশ সংরচনা {
    পূর্ণাঙ্ক active;
    বর্ণ output_log_file[512];
} Logger;

শূন্য logger_init(Logger *logger, ধ্রুব বর্ণ *log_dir, পূর্ণাঙ্ক process_rank, পূর্ণাঙ্ক resume) {
    // currently, only rank 0 writes logs
    logger->active = 0;
    যদি (log_dir != রিক্ত && process_rank == 0) {
        logger->active = 1;
        প্রতিজ্ঞা(strlen(log_dir) < 500); // being a bit lazy, could relax later
        snprintf(logger->output_log_file, 512, "%s/main.log", log_dir);
        যদি (resume == 0) {
            // wipe any existing logfile clean if we're starting fresh
            FILE *logfile = fopenCheck(logger->output_log_file, "w");
            fclose(logfile);
        }
    }
}

শূন্য logger_log_eval(Logger *logger, পূর্ণাঙ্ক step, ভিন্ন val) {
    যদি (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d eval:%.4f\n", step, val);
        fclose(logfile);
    }
}

শূন্য logger_log_val(Logger *logger, পূর্ণাঙ্ক step, ভিন্ন val_loss) {
    যদি (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d tel:%.4f\n", step, val_loss);
        fclose(logfile);
    }
}

শূন্য logger_log_train(Logger *logger, পূর্ণাঙ্ক step, ভিন্ন train_loss, ভিন্ন learning_rate, ভিন্ন grad_norm) {
    যদি (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d trl:%.4f lr:%.6f norm:%.2f\n", step, train_loss, learning_rate, grad_norm);
        fclose(logfile);
    }
}

#endif