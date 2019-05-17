from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
from six import iteritems
from subprocess import Popen, PIPE
from datetime import datetime

###################################################################

def save_model(ARGS, type, model_dir, model, log_file_path, epoch):
    if epoch % ARGS.model_save_interval == 0 or epoch == ARGS.epochs:
        save_name = os.path.join(model_dir, type + '_' + str(epoch) + '.pth')
        print_and_log(log_file_path, "Saving Model name: " + str(save_name))
        torch.save(model.state_dict(), save_name)        

def write_arguments_to_file(ARGS, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(ARGS)):
            f.write('%s: %s\n' % (key, str(value)))

def store_revision_info(src_path, output_dir, arg_string):
    try:
        # Get git hash
        cmd = ['git', 'rev-parse', 'HEAD']
        gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_hash = stdout.strip()
    except OSError as e:
        git_hash = ' '.join(cmd) + ': ' +  e.strerror
  
    try:
        # Get local changes
        cmd = ['git', 'diff', 'HEAD']
        gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_diff = stdout.strip()
    except OSError as e:
        git_diff = ' '.join(cmd) + ': ' +  e.strerror
    
    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        text_file.write('pytorch version: %s\n--------------------\n' % torch.__version__)  # @UndefinedVariable
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)

def print_and_log(log_file_path, string_to_write):
    print(string_to_write)
    with open(log_file_path, "a") as log_file:
        t = "[" + str(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')) + "] " 
        log_file.write(t + string_to_write + "\n")

###################################################################
