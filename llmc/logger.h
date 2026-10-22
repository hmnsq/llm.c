/*
Implements a simple logger that writes log files in the output directory.
The Logger object is stateless and uses append mode to write to log files.
*/
#నిర్వచితం_కాకపోతే LOGGER_H
#నిర్వచించు LOGGER_H

#అన్తర్భావ <assert.h>
#అన్తర్భావ <stdio.h>
#అన్తర్భావ <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#అన్తర్భావ "utils.h"

రకనిర్వచనం నిర్మాణం {
    పూర్ణాంకం చురుకైన;
    అక్షరం output_log_file[512];
} Logger;

శూన్య logger_init(Logger *logger, స్థిరరాశి అక్షరం *log_dir, పూర్ణాంకం process_rank, పూర్ణాంకం resume) {
    // currently, only rank 0 writes logs
    logger->చురుకైన = 0;
    ఒకవేళ (log_dir != ఖాళీ && process_rank == 0) {
        logger->చురుకైన = 1;
        ప్రతిజ్ఞా(అక్షరశ్రేణి_పొడవు(log_dir) < 500); // being a bit lazy, could relax later
        snprintf(logger->output_log_file, 512, "%s/main.log", log_dir);
        ఒకవేళ (resume == 0) {
            // wipe any existing logfile clean if we're starting fresh
            FILE *logfile = fopenCheck(logger->output_log_file, "w");
            fclose(logfile);
        }
    }
}

శూన్య logger_log_eval(Logger *logger, పూర్ణాంకం అడుగు, దశాంశం val) {
    ఒకవేళ (logger->చురుకైన == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d eval:%.4f\n", అడుగు, val);
        fclose(logfile);
    }
}

శూన్య logger_log_val(Logger *logger, పూర్ణాంకం అడుగు, దశాంశం val_loss) {
    ఒకవేళ (logger->చురుకైన == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d tel:%.4f\n", అడుగు, val_loss);
        fclose(logfile);
    }
}

శూన్య logger_log_train(Logger *logger, పూర్ణాంకం అడుగు, దశాంశం train_loss, దశాంశం learning_రేటు, దశాంశం grad_norm) {
    ఒకవేళ (logger->చురుకైన == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d trl:%.4f lr:%.6f norm:%.2f\n", అడుగు, train_loss, learning_రేటు, grad_norm);
        fclose(logfile);
    }
}

#అయితే_ముగింపు