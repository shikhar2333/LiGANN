## Instructions:
```
module load cuda/10.0
```
```
sinteractive -c 20 -g 2 -w gnodeXX
```
```
rsync -aP  ada:/share2/shubham.s/shubham /scratch
```
```
cd /scratch/shubham/crossdock_data
```
```
tar -xvf CrossDocked2020.tgz -C structs
```
```
jupyter-notebook --no-browser &
```
```
conda activate libmolgrid
```
```
bash ~/jp.sh <port on server> <port n local> <local ip> <user_name>
```

## Dependencies install:
```
conda env create -f environment.yml
```

## TO DO:
- [x] script to get the number of .gninatypes files for each of the docked structures.
- [x] get the sdf file for each of the docked protein structure.
- Script to generate the types file containing examples(.gninatypes files) used for training.
