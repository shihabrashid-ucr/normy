# NORMY: Non-uniform History Modeling for Open Retrieval Conversational Question Answering
Source code and dataset repository for our paper NORMY: Non-uniform History Modeling for Open Retrieval Conversational Question Answering. All source codes have been tested in Ubuntu-20.04 system with Python3.

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

## Experiments
First install all the requirements.
```pbash
pip3 install -r requirements.txt 
```
### PyLucene
We use [PyLucene](https://lucene.apache.org/pylucene/) to index our document collection. Follow these steps to install pylucene in your OS. It is better if the OS is clean.
1. Install openjdk-8:
`sudo apt install openjdk-8-jre openjdk-8-jdk openjdk-8-doc`
Ensure that you have ant installed, if you don't run `sudo apt install ant`. Note that if you had a different version of openjdk installed you need to either remove it or run `update-alternatives` so that version 1.8.0 is used.
2. Check that Java version is 1.8.0* with `java -version`
3. After installing openjdk-8 create a symlink:
```bash
cd /usr/lib/jvm
ln -s java-8-openjdk-amd64 java-8-oracle
```
4. Install python-dev: `sudo apt install python-dev`
5. Install JCC (in jcc subfolder of your pylucene folder):
```bash
python3 setup.py build
python3 setup.py install
```
6. Install pylucene (from the root of your pylucene folder). Edit Makefile, uncomment/edit the variables according to your setup.
```bash
PREFIX_PYTHON=/usr
ANT=ant
PYTHON=$(PREFIX_PYTHON)/bin/python3
JCC=$(PYTHON) -m jcc --shared
NUM_FILES=10
```
Then run
```bash
make
make test
sudo make install
```
7. If you see an error related to the shared mode of JCC remove `--shared` from Makefile.

### Install Pke
For keyphrase extraction we use open source library `pke`. To install
```bash
pipe install git+https://github.com/boudinfl/pke.git
```
pke relies on `spacy` (>= 3.2.3) for text processing
```bash
pip3 install -U spacy
python3 -m spacy download en_core_web_sm
```


