/*
Implements a simple logger that writes log files in the output directory.
The Logger object is stateless and uses append mode to write to log files.
*/
#اگر_غیر_تعریف_شدہ LOGGER_H
#تعریف_کرو LOGGER_H

#شامل <assert.h>
#شامل <stdio.h>
#شامل <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#شامل "utils.h"

قسمتعریف ساخت {
    صحیح فعال;
    حرف output_log_file[512];
} Logger;

خالی logger_init(Logger *logger, ثابت حرف *log_dir, صحیح process_rank, صحیح resume) {
    // currently, only rank 0 writes logs
    logger->فعال = 0;
    اگر (log_dir != تہی && process_rank == 0) {
        logger->فعال = 1;
        أكد(لڑی_لمبائی(log_dir) < 500); // being a bit lazy, could relax later
        snprintf(logger->output_log_file, 512, "%s/main.log", log_dir);
        اگر (resume == 0) {
            // wipe any existing logfile clean if we're starting fresh
            FILE *logfile = fopenCheck(logger->output_log_file, "w");
            fclose(logfile);
        }
    }
}

خالی logger_log_eval(Logger *logger, صحیح قدم, اعشاری val) {
    اگر (logger->فعال == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d eval:%.4f\n", قدم, val);
        fclose(logfile);
    }
}

خالی logger_log_val(Logger *logger, صحیح قدم, اعشاری val_loss) {
    اگر (logger->فعال == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d tel:%.4f\n", قدم, val_loss);
        fclose(logfile);
    }
}

خالی logger_log_train(Logger *logger, صحیح قدم, اعشاری train_loss, اعشاری learning_شرح, اعشاری grad_norm) {
    اگر (logger->فعال == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d trl:%.4f lr:%.6f norm:%.2f\n", قدم, train_loss, learning_شرح, grad_norm);
        fclose(logfile);
    }
}

#اگر_ختم