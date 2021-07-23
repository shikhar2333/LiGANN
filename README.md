## Instructions:
```
module load cuda/10.0
```
```
sinteractive -c 20 -g 2 -w gnodeXX
```
```
jupyter-notebook --no-browser &
```
```
conda activate libmolgrid
```
```
bash jp.sh <port on server> <port n local> <local ip> <user_name>
```

## Dependencies install:
```
conda env create -f environment.yml
```
