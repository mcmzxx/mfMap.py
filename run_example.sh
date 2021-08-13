cd /storage/zhang/mfMap.py/
Rscript gen_exmple_data.R
python3 mfMAP.py --patience 5 --label_fn dataset_labels.txt --input1_fn features_mut_cnv_comb.txt --input2_fn features_exp.txt --latent_space_dim 4 --organ COADREAD --p1_epoch_num 0 --separate_testing yes --use_cell yes --beta 0.5 --learning_rate 0.001 --level_4_dim 256 --level_3_dim_cnv 512 --level_3_dim_expr 512 --level_2_dim_cnv 1024 --level_2_dim_expr 1024 --classifier_1_dim 128 --classifier_2_dim 64;
