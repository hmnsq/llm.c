/*
Implements a simple logger that writes log files in the output directory.
The Logger object is stateless and uses append mode to write to log files.
*/
#ifndef LOGGER_H
#define LOGGER_H

#અન્તર્ભાવ <assert.h>
#અન્તર્ભાવ <stdio.h>
#અન્તર્ભાવ <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#અન્તર્ભાવ "utils.h"

પ્રકારવ્યાખ્યા સંરચના {
    પૂર્ણાંક active;
    અક્ષર output_log_file[512];
} Logger;

શૂન્ય logger_init(Logger *logger, અચલ અક્ષર *log_dir, પૂર્ણાંક process_rank, પૂર્ણાંક resume) {
    // currently, only rank 0 writes logs
    logger->active = 0;
    જો (log_dir != ખાલી && process_rank == 0) {
        logger->active = 1;
        પ્રતિજ્ઞા(strlen(log_dir) < 500); // being a bit lazy, could relax later
        snprintf(logger->output_log_file, 512, "%s/main.log", log_dir);
        જો (resume == 0) {
            // wipe any existing logfile clean if we're starting fresh
            FILE *logfile = fopenCheck(logger->output_log_file, "w");
            fclose(logfile);
        }
    }
}

શૂન્ય logger_log_eval(Logger *logger, પૂર્ણાંક step, દશાંશ val) {
    જો (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d eval:%.4f\n", step, val);
        fclose(logfile);
    }
}

શૂન્ય logger_log_val(Logger *logger, પૂર્ણાંક step, દશાંશ val_loss) {
    જો (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d tel:%.4f\n", step, val_loss);
        fclose(logfile);
    }
}

શૂન્ય logger_log_train(Logger *logger, પૂર્ણાંક step, દશાંશ train_loss, દશાંશ learning_rate, દશાંશ grad_norm) {
    જો (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d trl:%.4f lr:%.6f norm:%.2f\n", step, train_loss, learning_rate, grad_norm);
        fclose(logfile);
    }
}

#endif