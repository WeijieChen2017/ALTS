pip list
unzip ALTS.zip
unzip WORD.zip
rm WORD.zip
rm ALTS.zip
mv WORD ./ALTS/data_dir/
cd ALTS

# Run the Python script with the arguments passed to this shell script
python monai_train_v2_wb.py $@

# Use parameters to construct a unique filename for the output tar file
depth=$(echo $@ | grep -oP '(?<=--model_depth )\d+')
channels=$(echo $@ | grep -oP '(?<=--model_start_channels )\d+')
units=$(echo $@ | grep -oP '(?<=--model_num_res_units )\d+')
norm=$(echo $@ | grep -oP '(?<=--model_norm )[a-zA-Z]+')

tar_filename="proj_wb_depth${depth}_ch${channels}_units${units}_norm${norm}.tar.gz"

# Tar the project directory and name it uniquely
tar -czvf $tar_filename ALTS/proj_dir/WORD_base

# Cleanup
rm -rf ./data_dir/WORD
cd ..
rm -rf ALTS
