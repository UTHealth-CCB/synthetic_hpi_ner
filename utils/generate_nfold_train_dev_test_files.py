# -*- coding: utf-8 -*-

import os
import shutil
import random
import math
import copy

def get_file_list(path, ext, wo_ext=False):
    all_files = os.listdir(os.path.abspath(path))
    data_files = [data_file[:-len(ext)] if wo_ext else data_file for data_file in list(filter(lambda file: file.endswith(ext), all_files))]
    return data_files

def randomize_files(file_list):
    random.shuffle(file_list)

def get_train_dev_test_files(file_list, test_dev_rate, n_fold):
    if n_fold:
        # the test_dev_rate should be as 1/n_fold
        test_dev_rate = 1.0/n_fold
    test_index = math.ceil(len(file_list) * test_dev_rate)    
    test = file_list[:test_index]
    dev = file_list[test_index:test_index*2]
    train = file_list[test_index*2:]
    return train, dev, test

# generate n-fold file lists into train/dev/test separate files,
# also generate one file-list file contain all train/dev/test file lists with
# name as file_list_fold_X (X means fold number) and format as: file-name-with-path\t|[train|dev|test]
# src_dir: source file dir
# file_list_dir: generated file lists dir
# n_fold: nfold generation
# wo_ext: file name without ext, 
# copy_file_ext: copy file name with ext extension
# overwrite_file_list: if True, will overwrite previously generated nfold file lists, or do nothing
# file_list_wo_path: True means without path, False: with path
# test_dev_rate: if None, will generate test/dev rate as 1/n_fold
def generate_nfold_file_lists(src_dir, file_list_dir, n_fold, copy_file_ext, wo_ext, overwrite_file_list=True, file_list_wo_path=False, test_dev_rate=None):
    if not os.path.exists(file_list_dir):
        print('{} not exist, create it...'.format(file_list_dir))
        os.mkdir(file_list_dir)     
    src_files = get_file_list(src_dir, copy_file_ext, wo_ext)    
    # generate n_fold train, dev, and test file lists
    for i in range(n_fold):
        print('{}-fold: {}'.format(n_fold, i+1))
        random.seed(i+1)
        randomize_files(src_files)
        train, dev, test = get_train_dev_test_files(src_files, test_dev_rate, n_fold)
        #fold_subdir = 'fold_{}'.format(i+1)
        #fold_path = os.path.join(des_path, EXT[1:],  fold_subdir)
        #if not os.path.exists(fold_path):
        #    print('{} not exist, create it...'.format(des_ext_dir))
        #    os.mkdir(fold_path)
        test_fold_fn = 'fold_{}_test.txt'.format(i+1)
        test_fold_pfn = os.path.join(file_list_dir, test_fold_fn)
        dev_fold_fn = 'fold_{}_dev.txt'.format(i+1)
        dev_fold_pfn = os.path.join(file_list_dir, dev_fold_fn)
        train_fold_fn = 'fold_{}_train.txt'.format(i+1)
        train_fold_pfn = os.path.join(file_list_dir, train_fold_fn)
        # write train, dev and test folds list
        if train and (not os.path.exists(train_fold_pfn) or 
            (os.path.exists(train_fold_pfn) and overwrite_file_list)):
            train.sort()
            with open(train_fold_pfn, 'w') as train_file:
                train_file.writelines('{}\n'.format(item) for item in train)
                # process .ann&.txt in copy_nfold_train_dev_test_files procedure
                #if copy_file_ext == '.ann':
                #    # for .ann, also copy the corresponding .txt file name
                #    train_file.writelines('{}\n'.format(item.replace('.ann', '.txt')) for item in train)
                train_file.writelines('\n')
        if dev and (not os.path.exists(dev_fold_pfn) or 
            (os.path.exists(dev_fold_pfn) and overwrite_file_list)):
            dev.sort()
            with open(dev_fold_pfn, 'w') as dev_file:
                dev_file.writelines('{}\n'.format(item) for item in dev)
                # process .ann&.txt in copy_nfold_train_dev_test_files procedure
                #if copy_file_ext == '.ann':
                #    # for .ann, also copy the corresponding .txt file name
                #    dev_file.writelines('{}\n'.format(item.replace('.ann', '.txt')) for item in dev)
                dev_file.writelines('\n')
        if test and (not os.path.exists(test_fold_pfn) or 
            (os.path.exists(test_fold_pfn) and overwrite_file_list)):
            test.sort()
            with open(test_fold_pfn, 'w') as test_file:
                test_file.writelines('{}\n'.format(item) for item in test)
                # process .ann&.txt in copy_nfold_train_dev_test_files procedure
                #if copy_file_ext == '.ann':
                #    # for .ann, also copy the corresponding .txt file name
                #    test_file.writelines('{}\n'.format(item.replace('.ann', '.txt')) for item in test)
                test_file.writelines('\n')
        # write train/dev/test into file_list_pfn
        file_list_fn = 'file_list_fold_{}.txt'.format(i+1)
        file_list_pfn = os.path.join(file_list_dir, file_list_fn)
        if not os.path.exists(file_list_pfn):
            with open(file_list_pfn, 'w') as fl_wf:
                pfns = ['{}\ttrain'.format(fn if file_list_wo_path else os.path.join(src_dir, fn)) for fn in train]
                pfns += ['{}\tdev'.format(fn if file_list_wo_path else os.path.join(src_dir, fn)) for fn in dev]
                pfns += ['{}\ttest'.format(fn if file_list_wo_path else os.path.join(src_dir, fn)) for fn in test]
                # process .ann&.txt in copy_nfold_train_dev_test_files procedure
                #if copy_file_ext == '.ann':
                #    # for .ann, also copy the corresponding .txt file name
                #    pfns += ['{}\ttrain'.format(fn.replace('.ann', '.txt') if file_list_wo_path else os.path.join(src_dir, fn.replace('.ann', '.txt'))) for fn in train]    
                #    pfns += ['{}\tdev'.format(fn.replace('.ann', '.txt') if file_list_wo_path else os.path.join(src_dir, fn.replace('.ann', '.txt'))) for fn in dev]    
                #    pfns += ['{}\ttest'.format(fn.replace('.ann', '.txt') if file_list_wo_path else os.path.join(src_dir, fn.replace('.ann', '.txt'))) for fn in test]
                fl_wf.writelines('\n'.join(pfns))
                fl_wf.writelines('\n')
    

# copy files in filenames to des_dir, src_dir: if not None, join with filenames, or filenames should already contain path
def copy_files(filenames, des_dir, src_dir=None):
    if not os.path.exists(des_dir):
        print('{} not exists, create it...'.format(des_dir))
        os.mkdir(des_dir)    
    for fn in filenames:
        fname = os.path.basename(fn)
        des_pfn = os.path.join(des_dir, fname)
        pfn = os.path.join(src_dir, fn) if src_dir else fn
        if not os.path.exists(des_pfn):
            print('copy {} to {}'.format(pfn, des_pfn))
            try:
                shutil.copyfile(pfn, des_pfn)
            except Exception as e:
                if os.path.exists(des_pfn):
                    os.remove(des_pfn)
                raise RuntimeError('Exception {}: copy_fioes from {} to {}'.format(e, pfn, des_pfn))

# Copy n-fold files to each fold dir
# Suppose file_list_file name is as file_list_fold_X with X means the fold number
# and the format of file_list file is as: file-name-with-path\t[train|dev|test]
# copy_ext means file name ext in file_list
def copy_nfold_train_dev_test_files(file_list_dir, des_dir, n_fold, src_dir=None, copy_ext=None, file_list_wo_path=None, wo_ext=None):
    for i in range(n_fold):
        des_dir_new = os.path.join(des_dir, 'fold_{}'.format(i+1))
        if not os.path.exists(des_dir_new):
            print('{} not exists, create it...'.format(des_dir_new))
            os.mkdir(des_dir_new)                            
        train_dir = os.path.join(des_dir_new, 'train')
        if not os.path.exists(train_dir):
            print('{} not exists, create it...'.format(train_dir))
            os.mkdir(train_dir)            
        dev_dir = os.path.join(des_dir_new, 'dev')
        if not os.path.exists(dev_dir):
            print('{} not exists, create it...'.format(dev_dir))
            os.mkdir(dev_dir)
        test_dir = os.path.join(des_dir_new, 'test')
        if not os.path.exists(test_dir):
            print('{} not exists, create it...'.format(test_dir))
            os.mkdir(test_dir)
        file_dir = {'train': train_dir, 'dev': dev_dir, 'test': test_dir}
        file_list_pfn = os.path.join(file_list_dir, 'file_list_fold_{}.txt'.format(i+1))        
        file_list = {'train':[], 'dev':[], 'test':[]}
        with open(file_list_pfn, 'r') as fl_rf:            
            for line in fl_rf:
                line = line.strip('\n')
                if line:
                    if file_list_wo_path:
                        assert(src_dir)
                        src_pfn = os.path.join(src_dir, line.split('\t')[0])
                    else:
                        src_pfn = line.split('\t')[0]
                    if wo_ext:
                        assert(copy_ext)
                        src_pfn = src_pfn + copy_ext
                    train_dev_test = line.split('\t')[1]
                    file_list[train_dev_test].append(src_pfn)
        # copy train/dev/test files        
        for tdt in file_list.keys():
            copy_files(file_list[tdt], file_dir[tdt], src_dir)

def combine_files(filenames, out_file):
    with open(out_file, 'w', encoding='utf-8') as fp:
        # truncate file if exists
        pass    
    file_count = 0
    try:
        with open(out_file,'a', encoding='utf-8') as wfp:
            for pfn in filenames:  
                #print(filename)  
                file_count += 1    
                with open(pfn,'r', encoding='utf-8') as rfp:
                    shutil.copyfileobj(rfp, wfp)
    except Exception as e:
        #wfp.close()
        print('combine: exeption occured when combining {}!e:'.format(out_file, e))
        if os.path.exists(out_file):
            print('Since exception occured during combining {}, remove it...'.format(out_file))
            os.remove(out_file)

    print('Combined {} files to {}'.format(file_count, out_file))   

def combine_nfold_train_dev_test_files(file_list_dir, combined_data_dir, n_fold, src_dir=None, combine_ext='.bio', file_list_wo_path=None, wo_ext=None):
    for i in range(n_fold):
        des_dir = os.path.join(combined_data_dir, 'fold_{}'.format(i+1))
        if os.path.exists(des_dir):
            print('{} exists, not create it...'.format(des_dir))
            #continue
        else:
            print('{} not exists, create it...'.format(des_dir))
            os.mkdir(des_dir)
        des_input_dir = os.path.join(des_dir, 'input')
        if os.path.exists(des_input_dir):
            print('{} exists, not create it...'.format(des_input_dir))
        else:
            print('{} not exists, create it...'.format(des_input_dir))
            os.mkdir(des_input_dir)
        des_model_dir = os.path.join(des_dir, 'model')
        if os.path.exists(des_model_dir):
            print('{} exists, not create it...'.format(des_model_dir))
        else:
            print('{} not exists, create it...'.format(des_model_dir))
            os.mkdir(des_model_dir)
        des_output_dir = os.path.join(des_dir, 'output')
        if os.path.exists(des_output_dir):
            print('{} exists, not create it...'.format(des_output_dir))
        else:
            print('{} not exists, create it...'.format(des_output_dir))
            os.mkdir(des_output_dir)            
        file_list_pfn = os.path.join(file_list_dir, 'file_list_fold_{}.txt'.format(i+1))
        file_list = {'train':[], 'dev':[], 'test':[]}
        with open(file_list_pfn, 'r') as fl_rf:            
            for line in fl_rf:
                line = line.strip('\n')
                if line:
                    if file_list_wo_path:
                        assert(src_dir)
                        src_pfn = os.path.join(src_dir, line.split('\t')[0])
                    else:
                        src_pfn = line.split('\t')[0]
                    if wo_ext:
                        assert(combine_ext)
                        src_pfn = src_pfn + combine_ext                    
                    train_dev_test = line.split('\t')[1]
                    file_list[train_dev_test].append(src_pfn)
        # combine train/dev/test files
        ext = os.path.splitext(src_pfn)[1]
        for tdt in file_list.keys():
            out_file = os.path.join(des_input_dir, tdt+ext)
            if os.path.exists(out_file):
                print('{} exists, ignore it...'.format(out_file))
            else:
                print('{} not exists, combine and create it...'.format(out_file))
                combine_files(file_list[tdt], out_file) 
    
if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(cur_dir, '..')
    synth_hpi = {   
        'nfolds': {
            '10fold': {
                'n_fold': 10,
                'test_dev_rate': 0.1, # split rate for test and dev, train rate is 1-test_dev_rate*2
            }
        },                     
        'data': {
            # all the datasets in the 'data' key share the same nfold split strategy, support for different tokenized data
            'default': {
                'data_dir': os.path.join(root_dir, 'data/annotation/synth/'),
                #'generate_ext': '.bio', # used for generate_nfold_file_lists, deprecated, now set as the first item of file_ext_subfolds
                'file_ext_subfolds': {
                    '.bio': {'src_subfold': 'bio/all', 'copy_des_subfold': 'bio/{n_fold}fold', 'combine_des_subfold': 'bio/combined/{n_fold}fold', }, 
                    #'.xmi': {'src_subfold': 'xmi/all', 'copy_des_subfold': 'xmi/{n_fold}fold', 'combine_des_subfold': '', },                         
                },
            },
        },
        # file_list_subfold will be under the first key of data's data_dir
        'file_list_subfold': '{n_fold}fold',
        'overwrite_file_list': False, # if True, will overwrite previous generated file list files
        'file_list_wo_path': True, # without path in the file lists before file name        
        'wo_ext': True, # without ext in the n_fold file lists
    }
    datasets = {
        'synth': synth_hpi,
        #'ner_i2b2_hpi': ner_i2b2_hpi,
        #'ner_mtsamples_hpi': ner_mtsamples_hpi,
        #'ner_utnotes_hpi': ner_utnotes_hpi,         
    }    
    for dt in datasets:
        print(dt)
        nfolds = datasets[dt]['nfolds']
        data = datasets[dt]['data']
        overwrite_file_list = datasets[dt]['overwrite_file_list']
        file_list_subfold = datasets[dt]['file_list_subfold']
        file_list_wo_path = datasets[dt]['file_list_wo_path']
        wo_ext = datasets[dt]['wo_ext']
        for nfold in nfolds:
            n_fold = nfolds[nfold]['n_fold']
            test_dev_rate = nfolds[nfold]['test_dev_rate'] # test
            for sdt_idx, sdt in enumerate(data):
                data_dir = data[sdt]['data_dir']
                file_ext_subfolds = data[sdt]['file_ext_subfolds']
                if sdt_idx==0:
                    # generate file_list_dir under first item of data's data_dir
                    file_list_dir = os.path.join(data_dir, file_list_subfold.format(n_fold=n_fold))
                for ext_idx, ext in enumerate(file_ext_subfolds):
                    if ext_idx==0:
                        # 1. generate nfold file lists, only one time
                        # generate using first item of file_ext_subfolds
                        generate_src_dir = os.path.join(data_dir, file_ext_subfolds[ext]['src_subfold']) 
                        print('generate_nfold_file_lists: {} to {}'.format(generate_src_dir, file_list_dir))
                        generate_nfold_file_lists(
                            src_dir=generate_src_dir, 
                            file_list_dir=file_list_dir, 
                            n_fold=n_fold, 
                            copy_file_ext=ext, 
                            wo_ext=wo_ext, 
                            overwrite_file_list=overwrite_file_list,
                            file_list_wo_path=file_list_wo_path,
                            test_dev_rate=test_dev_rate
                        )
                    #2. combine train/dev/test files based on generated file-list-dir instead of copied files, do not combine .xmi or .ann files
                    if file_ext_subfolds[ext]['combine_des_subfold']: 
                        combine_src_dir = os.path.join(data_dir, file_ext_subfolds[ext]['src_subfold'])
                        combine_des_dir = os.path.join(data_dir, file_ext_subfolds[ext]['combine_des_subfold'].format(n_fold=n_fold))                
                        if not os.path.exists(combine_des_dir):
                            print('{} not exist, create it...'.format(combine_des_dir))
                            os.makedirs(combine_des_dir, exist_ok=True)
                        combine_nfold_train_dev_test_files(file_list_dir, combine_des_dir, n_fold, combine_src_dir, ext, file_list_wo_path, wo_ext)
                    # 3. copy nfold train/dev/test files                
                    copy_src_dir = os.path.join(data_dir, file_ext_subfolds[ext]['src_subfold'])                
                    copy_des_dir = os.path.join(data_dir, file_ext_subfolds[ext]['copy_des_subfold'].format(n_fold=n_fold))                
                    os.makedirs(copy_des_dir, exist_ok=True)
                    copy_nfold_train_dev_test_files(file_list_dir, copy_des_dir, n_fold, copy_src_dir, ext, file_list_wo_path, wo_ext)                   
    print('generate nfold train, dev and test lists done.')
