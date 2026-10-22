/*
Implements a simple logger that writes log files in the output directory.
The Logger object is stateless and uses append mode to write to log files.
*/
#ifndef LOGGER_H
#define LOGGER_H

#अन्तर्भाव <assert.h>
#अन्तर्भाव <stdio.h>
#अन्तर्भाव <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#अन्तर्भाव "utils.h"

प्रकारनिर्देश संरचना {
    पूर्णाङ्क active;
    वर्ण output_log_file[512];
} Logger;

शून्य logger_init(Logger *logger, ध्रुव वर्ण *log_dir, पूर्णाङ्क process_rank, पूर्णाङ्क resume) {
    // currently, only rank 0 writes logs
    logger->active = 0;
    जइ (log_dir != रिक्त && process_rank == 0) {
        logger->active = 1;
        प्रतिज्ञा(strlen(log_dir) < 500); // being a bit lazy, could relax later
        snprintf(logger->output_log_file, 512, "%s/main.log", log_dir);
        जइ (resume == 0) {
            // wipe any existing logfile clean if we're starting fresh
            FILE *logfile = fopenCheck(logger->output_log_file, "w");
            fclose(logfile);
        }
    }
}

शून्य logger_log_eval(Logger *logger, पूर्णाङ्क step, भिन्न val) {
    जइ (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d eval:%.4f\n", step, val);
        fclose(logfile);
    }
}

शून्य logger_log_val(Logger *logger, पूर्णाङ्क step, भिन्न val_loss) {
    जइ (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d tel:%.4f\n", step, val_loss);
        fclose(logfile);
    }
}

शून्य logger_log_train(Logger *logger, पूर्णाङ्क step, भिन्न train_loss, भिन्न learning_rate, भिन्न grad_norm) {
    जइ (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d trl:%.4f lr:%.6f norm:%.2f\n", step, train_loss, learning_rate, grad_norm);
        fclose(logfile);
    }
}

#endif