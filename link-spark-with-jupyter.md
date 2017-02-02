## Install jupyter with conda

```
conda install jupyter
```

## Get `jupyter binary executable path`

```
which jupyter
```

output

```
/Users/mingchen/anaconda2/bin/jupyter
```

## Link spark with jupyter

```
export PYSPARK_DRIVER_PYTHON=/Users/mingchen/anaconda2/bin/jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --NotebookApp.open_browser=False --NotebookApp.ip='*' --NotebookApp.port=8880"
```

You can also add the two environmental variables to the `~/.bash_profile` file to permenantly link spark with jupyter

## Run jupyter notebook

```
pyspark
```

Then go to [http://127.0.0.1:8880](http://127.0.0.1:8880)

