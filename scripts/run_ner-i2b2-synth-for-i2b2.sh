#!/bin/bash
set -x

# train ner: train using source train/dev data to predict target test data
root_dir=../ner
source_dataset=(i2b2 synth)
len_source_dt=${#source_dataset[@]}
target_dataset=(i2b2_synth_augment_for_i2b2_with_whole_synth)
len_target_dt=${#target_dataset[@]}
epoch=25
gpu=5
pytrain=$root_dir/chars_bilstm_crf/main.py
plevaluate=$root_dir/chars_bilstm_crf/conlleval
data_root_dir=$root_dir/data/ctg_hpi
glove_file=$root_dir/embedding/glove.840B.300d.txt
EXT='.bio'
delete_glove_preds_metrics=false
debug=false # true will only prepare the data without train and predict, false will prepare the data and do the train and predict
n_fold=10
for ((target_dt_idx=0; target_dt_idx<$len_target_dt; target_dt_idx++))
do
    for ((fold_idx=1; fold_idx<=$n_fold; fold_idx++)) 
    do          
        target_dataset_dir=$data_root_dir/${target_dataset[$target_dt_idx]}        
        if [ ! -d "$target_dataset_dir" ]; then
            echo "$target_dataset_dir not exists, create it..."
            mkdir $target_dataset_dir
        else
            echo "$target_dataset_dir exists, not create it..."
        fi
        target_fold_dir=$target_dataset_dir/fold_${fold_idx}
        if [ ! -d "$target_fold_dir" ]; then
            echo "$target_fold_dir not exists, create it..."
            mkdir $target_fold_dir                
        fi
        target_data_dir=$target_fold_dir/input
        if [ ! -d "$target_data_dir" ]; then
            echo "$target_data_dir not exists, create it..."
            mkdir $target_data_dir                
        fi
        target_output_dir=$target_fold_dir/output
        if [ ! -d "$target_output_dir" ]; then
            echo "$target_output_dir not exists, create it..."
            mkdir $target_output_dir
        fi
        # add epoch subfold
        target_output_dir=$target_fold_dir/output/epoch$epoch
        if [ ! -d "$target_output_dir" ]; then
            echo "$target_output_dir not exists, create it..."
            mkdir $target_output_dir
        fi
        target_model_dir=$target_fold_dir/model
        if [ ! -d "$target_model_dir" ]; then
            echo "$target_model_dir not exists, create it..."
            mkdir $target_model_dir
        fi
        # add epoch subfold
        target_model_dir=$target_fold_dir/model/epoch$epoch
        if [ ! -d "$target_model_dir" ]; then
            echo "$target_model_dir not exists, create it..."
            mkdir $target_model_dir
        fi
        log_file=$target_output_dir/${target_dataset}_epoch$epoch.log
        target_train_file=$target_data_dir/train$EXT
        target_dev_file=$target_data_dir/dev$EXT
        target_test_file=$target_data_dir/test$EXT

        glove_npz=$target_data_dir/glove.npz
        testb_preds_file=$target_output_dir/score/testb.preds.txt
        testb_metrics_file=$target_output_dir/score/score.testb.metrics.txt
        # 0. remove target input glove-related files
        # rm glove.npz, testb preds&metrics to re-train&evaluate
        if [ "$delete_glove_preds_metrics" == true ]; then
            if [ -d "$target_data_dir" ] && [ "$target_data_dir" != "" ] && [ "$target_data_dir" != "/" ]; then
                echo "delete $glove_npz, $testb_preds_file, and $testb_metrics_file"
                rm $target_data_dir/*.words.txt
                rm $target_data_dir/*.tags.txt
                rm $target_data_dir/*.chars.txt
                rm $glove_npz
                rm $testb_preds_file
                rm $testb_metrics_file
            fi
        fi
        #1. rm target model files
        if [ ! -f "$testb_preds_file" ]; then
            if [ -d "$target_model_dir" ] && [ "$target_model_dir" != "" ] && [ "$target_model_dir" != "/" ]; then
                echo "Not found $testb_preds_file, delete files on $target_model_dir"
                rm $target_model_dir/checkpoint
                rm $target_model_dir/events.out.*
                rm $target_model_dir/graph.pbtxt                
                rm $target_model_dir/model.ckpt*
                rm $target_model_dir/eval/events.out.*
            fi
        fi
        #2. rm target output files
        if [ ! -f "$testb_metrics_file" ]; then
            if [ -d "$target_output_dir" ] && [ "$target_output_dir" != "" ] && [ "$target_output_dir" != "/" ]; then
                echo "Not found $testb_metrics_file, delete files on $target_output_dir"
                rm $target_output_dir/*.log
                rm $target_output_dir/score/*.txt                
            fi
        fi
        first_dump=true
        for ((source_dt_idx=0; source_dt_idx<$len_source_dt; source_dt_idx++))
        do
            source_data_dir=$data_root_dir/${source_dataset[$source_dt_idx]}/fold_${fold_idx}/input
            source_train_file=$source_data_dir/train$EXT
            source_dev_file=$source_data_dir/dev$EXT
            source_test_file=$source_data_dir/test$EXT
            
            #3. copy source target train/dev/test files to target
            if [ ! -f "$target_train_file" ] || [ ! -f "$target_dev_file" ] || [ ! -f "$target_test_file" ]; then
                if [ "$first_dump" == true ]; then
                    #cp -rf $source_train_file $target_train_file
                    cat $source_train_file > $target_train_file
                    cat $source_dev_file >  $target_dev_file
                    cat $source_test_file > $target_test_file
                    first_dump=false
                else
                    cat $source_train_file >> $target_train_file
                    cat $source_dev_file >> $target_dev_file
                    # for jamia's paper to compare i2b2-synth-for-i2b2 with i2b2-for-i2b2, the following source (synth)'s test file will be used as target-train-file instead of target-test-file
                    #cat $source_test_file >> $target_test_file
                    cat $source_test_file >> $target_train_file
                fi
            fi
        done
        #4. prepocess input and generate embbeding file
        if [ ! -f "$glove_npz" ]; then  
            if [ "$debug" == true ]; then
                echo "just for debug:"
                echo "python $root_dir/data_process/convert_glamble_bio2guilalau_bio.py $target_data_dir $EXT" 
                echo "python $root_dir/data_process/build_vocab.py $target_data_dir            "
                echo "python $root_dir/data_process/build_glove.py $target_data_dir $glove_file"
            else
                python $root_dir/data_process/convert_glamble_bio2guilalau_bio.py $target_data_dir $EXT
                python $root_dir/data_process/build_vocab.py $target_data_dir            
                python $root_dir/data_process/build_glove.py $target_data_dir $glove_file
            fi
        else
            echo "$glove_npz exists, ignore it..."  
        fi  
        #5. run training         
        if [ ! -f "$testb_preds_file" ]; then            
            if [ "$debug" == true ]
            then
                echo "just for debug: python $pytrain --gpu=$gpu --data_dir=$target_data_dir --output_dir=$target_output_dir --model_dir=$target_model_dir --epoch=$epoch | tee $log_file"
            else
                python $pytrain --gpu=$gpu --data_dir=$target_data_dir --output_dir=$target_output_dir --model_dir=$target_model_dir --epoch=$epoch | tee $log_file
            fi
        else
            echo "$testb_preds_file exists, ignore it..."
        fi
        #6. evaluate        
        if [ ! -f "$testb_metrics_file" ]; then
            if [ $debug = true ]
            then                
                echo "just for debug: perl $plevaluate < $testb_preds_file  | tee $testb_metrics_file"
            else
                perl $plevaluate < $testb_preds_file | tee $testb_metrics_file
            fi
        else
            echo "$testb_metrics_file exists, ignore it..."
        fi 
        #4. evaluate train and dev
        testa_preds_file=$target_output_dir/score/testa.preds.txt
        testa_metrics_file=$target_output_dir/score/score.testa.metrics.txt
        if [ ! -f "$testa_metrics_file" ]; then
            perl $plevaluate < $target_output_dir/score/testa.preds.txt | tee $testa_metrics_file
        fi
        train_preds_file=$target_output_dir/score/train.preds.txt
        train_metrics_file=$target_output_dir/score/score.train.metrics.txt
        if [ ! -f "$train_metrics_file" ]; then
            perl $plevaluate < $target_output_dir/score/train.preds.txt | tee $train_metrics_file
        fi
    done   
done

date
echo "$0 done."
set +x
