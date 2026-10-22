/*
Implements a simple logger that writes log files in the output directory.
The Logger object is stateless and uses append mode to write to log files.
*/
#اگر_تعریف_نشده LOGGER_H
#تعریف_کن LOGGER_H

#ضمن <assert.h>
#ضمن <stdio.h>
#ضمن <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#ضمن "utils.h"

تعريفنوع بنية {
    صحيح فعال;
    حرف output_log_file[512];
} Logger;

فراغ logger_init(Logger *logger, ثابت حرف *log_dir, صحيح process_rank, صحيح resume) {
    // currently, only rank 0 writes logs
    logger->فعال = 0;
    إذا (log_dir != عدم && process_rank == 0) {
        logger->فعال = 1;
        أكد(طول_رشته(log_dir) < 500); // being a bit lazy, could relax later
        snprintf(logger->output_log_file, 512, "%s/main.log", log_dir);
        إذا (resume == 0) {
            // wipe any existing logfile clean if we're starting fresh
            FILE *logfile = fopenCheck(logger->output_log_file, "w");
            fclose(logfile);
        }
    }
}

فراغ logger_log_eval(Logger *logger, صحيح گام, عائم val) {
    إذا (logger->فعال == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d eval:%.4f\n", گام, val);
        fclose(logfile);
    }
}

فراغ logger_log_val(Logger *logger, صحيح گام, عائم val_loss) {
    إذا (logger->فعال == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d tel:%.4f\n", گام, val_loss);
        fclose(logfile);
    }
}

فراغ logger_log_train(Logger *logger, صحيح گام, عائم train_loss, عائم learning_نرخ, عائم grad_norm) {
    إذا (logger->فعال == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d trl:%.4f lr:%.6f norm:%.2f\n", گام, train_loss, learning_نرخ, grad_norm);
        fclose(logfile);
    }
}

#پایان_اگر