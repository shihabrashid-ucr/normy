# NORMY: Non-uniform History Modeling for Open Retrieval Conversational Question Answering
Source code and dataset repository for our paper NORMY: Non-uniform History Modeling for Open Retrieval Conversational Question Answering. All source codes have been tested in Ubuntu-20.04 system.

Download/clone the repository and `cd` into project directory.

## Datasets
We have used three datasets for our experiments.

1. OrConvQA

Download Wikipedia corpus [here](https://ciir.cs.umass.edu/downloads/ORConvQA/all_blocks.txt.gz) and unzip the `gz` file with `gzip -d all_blocks.txt.gz`. Put the `all_blocks.txt` file under `datasets/orqa/`. The conversation file is already under this subdirectory.

2. doc2dial

We have contributed to extend doc2dial dataset for open retrieval task. We release our data [here](https://figshare.com/s/bf5ba94bc71b31fffdf2). doc2dial original data source from governemnt websites and our appended Wikipedia corpus can be found in `doc2dial_with_wiki_dataset.json` and the test file in OrConvQA format can be found in `doc2dial_validation_orconvqa_format_with_pos_cntxt.json`. Put both files under `datasets/doc2dial/`.

3. ConvMix

To dowload their Wikipedia corpus follow these steps:
```bash
wget http://qa.mpi-inf.mpg.de/convinse/convmix_data/wikipedia.zip
unzip wikipedia.zip -d datasets/convmix/
rm wikipedia.zip
```
This will download `wikipedia_dump.pickle` under `datasets/convmix/`. The conversation file `train_set_ALL.json` is already under this subdirectory. 
