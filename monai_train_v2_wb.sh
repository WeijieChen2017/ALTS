# docker build -t convez376/monai_with_wb:latest .
# docker run --gpus all -ti -v ./:/wchen376 --ipc=host convez376/monai_with_wb:latest
pip list
# cd /wchen376
# unzip ALTS.zip
# unzip WORD.zip
# rm WORD.zip
# rm ALTS.zip
# mv WORD ./ALTS/data_dir/
# cd ALTS
# python monai_train_v2_wb.py
# rm -rf ./data_dir/WORD
# cd ../
# tar -czvf proj.tar.gz ALTS/proj_dir/WORD_base
# rm -rf ALTS
