# synthetic_hpi_ner
This repository contains the manually annotated synthetic corpus and codes for the paper: Are Synthetic Clinical Notes Useful for Real Natural Language Processing Tasks: A Case Study on Clinical Entity Recognition.
  
Requirements

Install python, tensorflow. We use Python 3.6, Tensorflow 1.15.2.
If you plan to use GPU computation, install CUDA.

Train all the synthetic language models using the scripts (run_train_#.sh/run_gen_#.sh) in each model-folder under synth_hpi with I2B2 2010 and N2C2 2018 HPI (History of Present Illness) section data. The original I2B2 2010 and N2C2 2018 challenges data are available at https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/ which would require additional login or register. 

Train all the NER models using the scripts in scripts folder. The manuually annotated synthetic data are available at data/annotation/synth. Before training the 10-fold NER models, use the python file generate_nfold_train_dev_test_files.py at utils to generate 10 fold train/dev/test files.

Once finished training, the estimated scores can be found in the output folder of each training data.
