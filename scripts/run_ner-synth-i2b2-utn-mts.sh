#!/bin/bash
set -x
date

# train ner: train using source train/dev data to predict target test data
root_dir=../ner
dataset=(synth i2b2 utnotes mtsamples) # utnotes_hpi i2b2_hpi mtsamples_hpi)
len_dt=${#dataset[@]}
epoch=200
gpu=4
pytrain=$root_dir/chars_bilstm_crf/main.py
plevaluate=$root_dir/chars_bilstm_crf/conlleval
data_root_dir=$root_dir/data/ctg_hpi
glove_file=$root_dir/embedding/glove.840B.300d.txt
EXT='.bio'
n_fold=10
# if delete_glove_preds_metrics is true
# will delete all glove related preprocessed files and preds&eval-metrics files
# then will also delete all models' files
delete_glove_preds_metrics=false 
# if preprocess true, generating proprocessed file like glove.npz, etc.
preprocess=true 
train=true
evaluate=true
for ((dt_idx=0; dt_idx<$len_dt; dt_idx++))    
do           
    for ((fold_idx=1; fold_idx<=$n_fold; fold_idx++)) 
    do  
        fold_dir=$data_root_dir/${dataset[$dt_idx]}/fold_${fold_idx}        
        data_dir=$fold_dir/input
        train_file=$data_dir/train$EXT
        dev_file=$data_dir/dev$EXT
        test_file=$data_dir/test$EXT            
        output_dir=$fold_dir/output
        if [ ! -d "$output_dir" ]; then
            echo "$output_dir not exists, create it..."
            mkdir $output_dir
        fi
        # add epoch subfold
        output_dir=$fold_dir/output/epoch$epoch
        if [ ! -d "$output_dir" ]; then
            echo "$output_dir not exists, create it..."
            mkdir $output_dir
        fi
        model_dir=$fold_dir/model
        if [ ! -d "$model_dir" ]; then
            echo "$model_dir not exists, create it..."
            mkdir $model_dir
        fi
        # add epoch subfold
        model_dir=$fold_dir/model/epoch$epoch
        if [ ! -d "$model_dir" ]; then
            echo "$model_dir not exists, create it..."
            mkdir $model_dir
        fi
        log_file=$output_dir/${dataset[$dt_idx]}_epoch$epoch.log
        glove_npz=$data_dir/glove.npz
        testb_preds_file=$output_dir/score/testb.preds.txt
        testb_metrics_file=$output_dir/score/score.testb.metrics.txt        
        # rm glove.npz, testb preds&metrics to re-train&evaluate
        if [ "$delete_glove_preds_metrics" == true ]; then
            echo "delete $glove_npz, $testb_preds_file, and $testb_metrics_file"
            rm $glove_npz
            rm $data_dir/*.tags.txt
            rm $data_dir/*.words.txt
            rm $data_dir/*.chars.txt
            rm $data_dir/params.json
            rm $testb_preds_file
            rm $testb_metrics_file
        fi
        #0.1. rm target model files
        if [ ! -f "$testb_preds_file" ]; then
            if [ -d "$model_dir" ] && [ "$model_dir" != "" ] && [ "$model_dir" != "/" ]; then
                echo 'Not found $testb_preds_file, delete files on $model_dir'            
                rm $model_dir/checkpoint
                rm $model_dir/events.out.*
                rm $model_dir/graph.pbtxt                
                rm $model_dir/model.ckpt*
                rm $model_dir/eval/events.out.*
            fi
        fi
        #2. rm target output files
        if [ ! -f "$testb_metrics_file" ]; then
            if [ -d "$output_dir" ] && [ "$output_dir" != "" ] && [ "$output_dir" != "/" ]; then
                echo "Not found $testb_metrics_file, delete files on $output_dir"
                rm $output_dir/*.log
                rm $output_dir/score/*.txt
            fi
        fi

        #1. prepocess input and generate embbeding file
        if [ "$preprocess" == true ]; then
            if [ ! -f "$glove_npz" ]; then
                python $root_dir/data_process/convert_glamble_bio2guilalau_bio.py $data_dir $EXT
                python $root_dir/data_process/build_vocab.py $data_dir
                python $root_dir/data_process/build_glove.py $data_dir $glove_file
            else
                echo "$glove_npz exists, ignore it..."  
            fi 
        fi
        #2. run training    
        if [ "$train" == true ]; then     
            if [ ! -f "$testb_preds_file" ]; then
                python $pytrain --gpu=$gpu --data_dir=$data_dir --output_dir=$output_dir --model_dir=$model_dir --epoch=$epoch | tee $log_file
            fi
        fi
        #3. evaluate        
        if [ "$evaluate" == true ]; then
            if [ ! -f "$testb_metrics_file" ]; then
                perl $plevaluate < $output_dir/score/testb.preds.txt | tee $testb_metrics_file
            fi                
            #4. evaluate train and dev
            testa_preds_file=$output_dir/score/testa.preds.txt
            testa_metrics_file=$output_dir/score/score.testa.metrics.txt
            if [ ! -f "$testa_metrics_file" ]; then
                perl $plevaluate < $output_dir/score/testa.preds.txt | tee $testa_metrics_file
            fi
            train_preds_file=$output_dir/score/train.preds.txt
            train_metrics_file=$output_dir/score/score.train.metrics.txt        
            if [ ! -f "$train_metrics_file" ]; then
                perl $plevaluate < $output_dir/score/train.preds.txt | tee $train_metrics_file
            fi
        fi
    done                
done

date
echo "$0 done."

set +x
