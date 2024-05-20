unzip ALTS.zip
unzip WORD.zip
rm WORD.zip
rm ALTS.zip
mv WORD ./ALTS/data_dir/
cd ALTS
python monai_train_v1.py
rm -rf ./data_dir/WORD
cd ../
zip -r ALTS_v1.zip ALTS
rm -rf ALTS
