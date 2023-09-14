sudo rm -rf data/data_dst/*.png
sudo rm -rf data/data_dst/*.png
sudo rm -rf data/data_dst/*.jpg
sudo rm -rf data/data_dst/*.jpg
sudo rm -rf data/data_dst/aligned/*
sudo rm -rf data/data_dst/aligned_debug/*
sudo rm -rf data/data_dst/merged/*
sudo rm -rf data/data_dst/merged_mask/*
sudo rm -rf data/data_src/*.png
sudo rm -rf data/data_src/*.jpg
sudo rm -rf data/data_src/aligned/*
sudo rm -rf data/data_src/aligned_debug/*

find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$|\.jpg$)" | xargs rm -rf