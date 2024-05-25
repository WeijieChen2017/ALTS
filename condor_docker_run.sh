docker build -t convez376/monai_with_wb:latest .
docker run --gpus all -ti -v ./:/wchen376 --ipc=host convez376/monai_with_wb:latest
pip list
unzip ALTS.zip
cd ALTS.zip
ls
