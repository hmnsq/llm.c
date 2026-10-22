/*
Implements a simple logger that writes log files in the output directory.
The Logger object is stateless and uses append mode to write to log files.
*/
#अगर_अघोषित LOGGER_H
#घोषणा LOGGER_H

#समावेश <assert.h>
#समावेश <stdio.h>
#समावेश <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#समावेश "utils.h"

प्रकार काष्ठा {
    पूर्णांक active;
    अक्षर output_log_file[512];
} Logger;

व्योम logger_init(Logger *logger, स्थिर अक्षर *log_dir, पूर्णांक process_rank, पूर्णांक resume) {
    // currently, only rank 0 writes logs
    logger->active = 0;
    अगर (log_dir != शून्य && process_rank == 0) {
        logger->active = 1;
        निश्चित(म_माप(log_dir) < 500); // being a bit lazy, could relax later
        snprintf(logger->output_log_file, 512, "%s/main.log", log_dir);
        अगर (resume == 0) {
            // wipe any existing logfile clean if we're starting fresh
            खाता *logfile = fopenCheck(logger->output_log_file, "w");
            ख_बंद(logfile);
        }
    }
}

व्योम logger_log_eval(Logger *logger, पूर्णांक step, भग्न val) {
    अगर (logger->active == 1) {
        खाता *logfile = fopenCheck(logger->output_log_file, "a");
        ख_लिखो(logfile, "s:%d eval:%.4f\n", step, val);
        ख_बंद(logfile);
    }
}

व्योम logger_log_val(Logger *logger, पूर्णांक step, भग्न val_loss) {
    अगर (logger->active == 1) {
        खाता *logfile = fopenCheck(logger->output_log_file, "a");
        ख_लिखो(logfile, "s:%d tel:%.4f\n", step, val_loss);
        ख_बंद(logfile);
    }
}

व्योम logger_log_train(Logger *logger, पूर्णांक step, भग्न train_loss, भग्न learning_rate, भग्न grad_norm) {
    अगर (logger->active == 1) {
        खाता *logfile = fopenCheck(logger->output_log_file, "a");
        ख_लिखो(logfile, "s:%d trl:%.4f lr:%.6f norm:%.2f\n", step, train_loss, learning_rate, grad_norm);
        ख_बंद(logfile);
    }
}

#पूर्ण_अगर