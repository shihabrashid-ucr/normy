import logging, sys
logging.disable(sys.maxsize)
import json
import pickle
import lucene
import os
from org.apache.lucene.store import MMapDirectory, SimpleFSDirectory, NIOFSDirectory
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions, DirectoryReader
from org.apache.lucene.search import IndexSearcher, BoostQuery, Query
from org.apache.lucene.search.similarities import BM25Similarity

document = []
with open('datasets/orqa/all_blocks.txt') as f:
    for line in f:
        document.append(json.loads(line))

def create_index(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    store = SimpleFSDirectory(Paths.get(dir))
    analyzer = StandardAnalyzer()
    config = IndexWriterConfig(analyzer)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    writer = IndexWriter(store, config)

    metaType = FieldType()
    metaType.setStored(True)
    metaType.setTokenized(False)

    contextType = FieldType()
    contextType.setStored(True)
    contextType.setTokenized(True)
    contextType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

    for k in range(0, len(document)):
        title = document[k]['id']
        content = document[k]['text']
        doc = Document()
        doc.add(Field('Title', str(title), metaType))
        doc.add(Field('Context', str(content), contextType))
        writer.addDocument(doc)
    writer.close()
    

lucene.initVM(vmargs=['-Djava.awt.headless=true'])
create_index('orqa_lucene_index/')



