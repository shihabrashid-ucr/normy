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

    with open('datasets/doc2dial/doc2dial_with_wiki_dataset.json') as outf:
        corpus_dicc = json.load(outf)
    print(f'json file loaded')
    i = 0
    for corp in corpus_dicc:
        print(f'i: {i}')
        content = corp['text']
        title = corp['doc_id']
        doc = Document()
        doc.add(Field('Title', str(title), metaType))
        doc.add(Field('Context', str(content), contextType))
        writer.addDocument(doc)
        i += 1
    writer.close()



lucene.initVM(vmargs=['-Djava.awt.headless=true'])
create_index('doc2dial_lucene_index/')