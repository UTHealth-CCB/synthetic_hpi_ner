#!/bin/bash
set -x
date

# train ner: train using source train/dev data to predict target test data
root_dir=../ner
source_dataset=(synth)
len_source_dt=${#source_dataset[@]}
target_dataset=(utnotes mtsamples i2b2) # i2b2_n2c2_synth) # mtsamples_hpi i2b2_n2c2_synth)
len_target_dt=${#target_dataset[@]}
epoch=25
gpu=0
pytrain=$root_dir/chars_bilstm_crf/main.py
pypredict=$root_dir/chars_bilstm_crf/predict.py
plevaluate=$root_dir/chars_bilstm_crf/conlleval
data_root_dir=$root_dir/data/ctg_hpi
glove_file=$root_dir/embedding/glove.840B.300d.txt
EXT='.bio'
n_fold=10
# if delete_glove_preds_metrics is true (prefered), 
# will delete glove related preprocessed files, and preds/metrics files
delete_glove_preds_metrics=false
# if preprocess true and no glove.npz file exists, 
# generating proprocessed file like glove.npz, etc.
preprocess=true
train=true
evaluate=true
for ((source_dt_idx=0; source_dt_idx<$len_source_dt; source_dt_idx++))
do           
    for ((target_dt_idx=0; target_dt_idx<$len_target_dt; target_dt_idx++))
    do
        for ((target_fold_idx=1; target_fold_idx<=$n_fold; target_fold_idx++))
        do          
            # Predict each fold of target test using 10 folds of source's train+test data as train and dev data as dev of re-trained source models
            for ((source_fold_idx=1; source_fold_idx<=$n_fold; source_fold_idx++))
            do                          
                source_data_dir=$data_root_dir/${source_dataset[$source_dt_idx]}/fold_${source_fold_idx}/input
                source_model_dir=$data_root_dir/${source_dataset[$source_dt_idx]}/fold_${source_fold_idx}/model/epoch$epoch
                source_train_file=$source_data_dir/train$EXT
                source_dev_file=$source_data_dir/dev$EXT
                source_test_file=$source_data_dir/test$EXT
                target_data_dir=$data_root_dir/${target_dataset[$target_dt_idx]}/fold_${target_fold_idx}/input
                target_test_file=$target_data_dir/test$EXT
                
                new_dataset=${source_dataset[$source_dt_idx]}_for_${target_dataset[$target_dt_idx]}_with_source_test_voting
                new_dataset_dir=$data_root_dir/$new_dataset 
                if [ ! -d "$new_dataset_dir" ]; then
                    #echo "$new_dataset_dir not exists, create it..."
                    mkdir $new_dataset_dir
                else
                    echo "$new_dataset_dir exists, not create it..."
                fi
                new_fold_dir=$new_dataset_dir/fold_${target_fold_idx}
                if [ ! -d "$new_fold_dir" ]; then
                    #echo "$new_fold_dir not exists, create it..."
                    mkdir $new_fold_dir                
                fi
                new_data_dir=$new_fold_dir/input
                if [ ! -d "$new_data_dir" ]; then
                    #echo "$new_data_dir not exists, create it..."
                    mkdir $new_data_dir                
                fi
                # add source-fold under epoch subfold
                new_data_dir=$new_fold_dir/input/source_fold_${source_fold_idx}
                if [ ! -d "$new_data_dir" ]; then
                    #echo "$new_data_dir not exists, create it..."
                    mkdir $new_data_dir
                fi                
                new_model_dir=$new_fold_dir/model
                if [ ! -d "$new_model_dir" ]; then
                    #echo "$new_model_dir not exists, create it..."
                    mkdir $new_model_dir
                fi
                # add epoch subfold
                new_model_dir=$new_fold_dir/model/epoch$epoch
                if [ ! -d "$new_model_dir" ]; then
                    #echo "$new_model_dir not exists, create it..."
                    mkdir $new_model_dir
                fi
                # add source-fold under epoch subfold
                new_model_dir=$new_fold_dir/model/epoch$epoch/source_fold_${source_fold_idx}
                if [ ! -d "$new_model_dir" ]; then
                    #echo "$new_model_dir not exists, create it..."
                    mkdir $new_model_dir
                fi
                new_output_dir=$new_fold_dir/output
                if [ ! -d "$new_output_dir" ]; then
                    #echo "$new_output_dir not exists, create it..."
                    mkdir $new_output_dir
                fi
                # add epoch subfold
                new_output_dir=$new_fold_dir/output/epoch$epoch
                if [ ! -d "$new_output_dir" ]; then
                    #echo "$new_output_dir not exists, create it..."
                    mkdir $new_output_dir
                fi
                # add source-fold under epoch subfold
                new_output_dir=$new_fold_dir/output/epoch$epoch/source_fold_${source_fold_idx}
                if [ ! -d "$new_output_dir" ]; then
                    #echo "$new_output_dir not exists, create it..."
                    mkdir $new_output_dir
                fi
                log_file=$new_output_dir/${new_dataset}_epoch$epoch_source_fold_${source_fold_idx}.log
                new_train_file=$new_data_dir/train$EXT
                new_dev_file=$new_data_dir/dev$EXT            
                new_test_file=$new_data_dir/test$EXT
                new_glove_npz=$new_data_dir/glove.npz
                new_testb_preds_file=$new_output_dir/score/testb.preds.txt
                new_testb_metrics_file=$new_output_dir/score/score.testb.metrics.txt
                # 0. remove target input glove-related files, 
                # rm glove.npz, testb preds&metrics to re-train&evaluate
                if [ "$delete_glove_preds_metrics" == true ]; then
                    if [ -d "$new_data_dir" ] && [ "$new_data_dir" != "" ] && [ "$new_data_dir" != "/" ]; then
                        #echo "delete $new_glove_npz, $new_testb_preds_file, and $new_testb_metrics_file"
                        rm $new_data_dir/*.words.txt
                        rm $new_data_dir/*.tags.txt
                        rm $new_data_dir/*.chars.txt
                        rm $new_glove_npz
                        rm $new_testb_preds_file
                        rm $new_testb_metrics_file
                    fi
                fi
                #1. rm new target model files
                if [ ! -f "$new_testb_preds_file" ]; then
                    if [ -d "$new_model_dir" ] && [ "$new_model_dir" != "" ] && [ "$new_model_dir" != "/" ]; then
                        #echo "Not found $new_testb_preds_file, delete files on $new_model_dir"
                        rm $new_model_dir/checkpoint
                        rm $new_model_dir/events.out.*
                        rm $new_model_dir/graph.pbtxt                
                        rm $new_model_dir/model.ckpt*
                        rm $new_model_dir/eval/events.out.*
                    fi
                fi
                #2. rm new target output files
                if [ ! -f "$new_testb_metrics_file" ]; then
                    if [ -d "$new_output_dir" ] && [ "$new_output_dir" != "" ] && [ "$new_output_dir" != "/" ]; then
                        #echo "Not found $new_testb_metrics_file, delete files on $new_output_dir"
                        rm $new_output_dir/*.log
                        rm $new_output_dir/score/*.txt
                    fi
                fi
                #3. copy source's train+test/dev and target's test files to new target
                if [ ! -f "$new_train_file" ]; then
                    #cp -rf $source_train_file $new_train_file
                    #to combine source train+test as new train file
                    cat $source_train_file > $new_train_file
                    cat $source_test_file >> $new_train_file
                else
                    echo "$new_train_file exists, not cp..."
                fi
                if [ ! -f "$new_dev_file" ]; then
                    #echo "cp -rf $source_dev_file $new_dev_file"
                    cp -rf $source_dev_file $new_dev_file
                else
                    echo "$new_dev_file exists, not cp..."
                fi            
                if [ ! -f "$new_test_file" ]; then
                    #echo "cp -rf $target_test_file $new_test_file"
                    cp -rf $target_test_file $new_test_file
                else
                    echo "$new_test_file exists, not cp..."
                fi   
                #4. prepocess input and generate embbeding file
                if [ "$preprocess" == "true" ]; then
                    if [ ! -f "$new_glove_npz" ]; then                
                        # regenerate input words/tags/vocab and embedding file
                        python $root_dir/data_process/convert_glamble_bio2guilalau_bio.py $new_data_dir $EXT
                        python $root_dir/data_process/build_vocab.py $new_data_dir
                        python $root_dir/data_process/build_glove.py $new_data_dir $glove_file
                    else
                        echo "$new_glove_npz exists, ignore it..."  
                    fi           
                fi                 
                #5. train/predict
                if [ "$train" == "true" ]; then
                    if [ ! -f "$new_testb_preds_file" ]; then
                        python $pytrain --gpu=$gpu --data_dir=$new_data_dir --output_dir=$new_output_dir --model_dir=$new_model_dir --epoch=$epoch | tee $log_file
                    else
                        echo "$new_testb_preds_file exists, ignore it..."
                    fi
                fi
                #6. evaluate            
                if [ "$evaluate" == "true" ]; then
                    if [ ! -f "$new_testb_metrics_file" ]; then
                        perl $plevaluate < $new_testb_preds_file | tee $new_testb_metrics_file
                    else
                        echo "$new_testb_metrics_file exists, ignore it..."
                    fi
                    new_testa_preds_file=$new_output_dir/score/testa.preds.txt
                    new_testa_metrics_file=$new_output_dir/score/score.testa.metrics.txt
                    if [ ! -f "$new_testa_metrics_file" ]; then
                        perl $plevaluate < $new_testa_preds_file | tee $new_testa_metrics_file
                    else
                        echo "$new_testa_metrics_file exists, ignore it..."
                    fi
                    new_train_preds_file=$new_output_dir/score/train.preds.txt
                    new_train_metrics_file=$new_output_dir/score/score.train.metrics.txt
                    if [ ! -f "$new_train_metrics_file" ]; then
                        perl $plevaluate < $new_train_preds_file | tee $new_train_metrics_file
                    else
                        echo "$new_train_metrics_file exists, ignore it..."
                    fi
                fi
            done
        done
    done
done

date
echo "$0 done."

set +x