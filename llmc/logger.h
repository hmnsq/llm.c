/*
Implements a simple logger that writes log files in the output directory.
The Logger object is stateless and uses append mode to write to log files.
*/
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ LOGGER_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ LOGGER_H

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#ਅਨ੍ਤਰ੍ਭਾਵ "utils.h"

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਪੂਰਨ ਸਰਗਰਮ;
    ਅੱਖਰ output_log_file[512];
} Logger;

ਖਾਲੀ logger_init(Logger *logger, ਅਚਲ ਅੱਖਰ *log_dir, ਪੂਰਨ process_rank, ਪੂਰਨ resume) {
    // currently, only rank 0 writes logs
    logger->ਸਰਗਰਮ = 0;
    ਜੇ (log_dir != ਨਿਰਾ && process_rank == 0) {
        logger->ਸਰਗਰਮ = 1;
        ਪ੍ਰਤਿਜ੍ਞਾ(ਲੜੀ_ਲੰਬਾਈ(log_dir) < 500); // being a bit lazy, could relax later
        snprintf(logger->output_log_file, 512, "%s/main.log", log_dir);
        ਜੇ (resume == 0) {
            // wipe any existing logfile clean if we're starting fresh
            FILE *logfile = fopenCheck(logger->output_log_file, "w");
            fclose(logfile);
        }
    }
}

ਖਾਲੀ logger_log_eval(Logger *logger, ਪੂਰਨ ਕਦਮ, ਦਸ਼ਮਲਵ val) {
    ਜੇ (logger->ਸਰਗਰਮ == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d eval:%.4f\n", ਕਦਮ, val);
        fclose(logfile);
    }
}

ਖਾਲੀ logger_log_val(Logger *logger, ਪੂਰਨ ਕਦਮ, ਦਸ਼ਮਲਵ val_loss) {
    ਜੇ (logger->ਸਰਗਰਮ == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d tel:%.4f\n", ਕਦਮ, val_loss);
        fclose(logfile);
    }
}

ਖਾਲੀ logger_log_train(Logger *logger, ਪੂਰਨ ਕਦਮ, ਦਸ਼ਮਲਵ train_loss, ਦਸ਼ਮਲਵ learning_ਦਰ, ਦਸ਼ਮਲਵ grad_norm) {
    ਜੇ (logger->ਸਰਗਰਮ == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d trl:%.4f lr:%.6f norm:%.2f\n", ਕਦਮ, train_loss, learning_ਦਰ, grad_norm);
        fclose(logfile);
    }
}

#ਜੇ_ਅੰਤ