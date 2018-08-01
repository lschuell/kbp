# from lxml import etree #for parser
import xml.etree.ElementTree as etree
from io import BytesIO
import pickle

from gensim.models import KeyedVectors
from gensim.test.utils import get_tmpfile
import os


# a mapping is (source, target, relation, confidence) or a list or anything with is iterable
# onto is (id, url, formalismName, formalismURI)

# serialization

def __get_ontology_string(onto, name):
    ontoString = ''
    if onto == None:
        return ontoString
    if len(onto) > 0:
        ontoString += '  <' + name + '>\n'
        ontoString += '    <Ontology rdf:about="' + onto[0] + '">\n'
        if len(onto) > 1:
            ontoString += '      <location>' + onto[1] + '</location>\n'
            if len(onto) > 3:
                ontoString += '      <formalism>\n'
                ontoString += '        <Formalism align:name="' + onto[2] + '" align:uri="' + onto[3] + '"/>\n'
                ontoString += '      </formalism>\n'
        ontoString += '    </Ontology>\n'
        ontoString += '  </' + name + '>\n'
    return ontoString


def __get_extension_string(extension):
    ext_string = ''
    if extension is None:
        return ext_string
    for key, value in extension.items():
        ext_string += "  <" + key + ">" + value + "</" + key + ">\n"
    return ext_string


def __get_mappings_string(alignments):
    mapString = ''
    if alignments == None:
        return mapString
    for source, target, relation, confidence in alignments:
        mapString += '  <map>\n'
        mapString += '    <Cell>\n'
        mapString += '      <entity1 rdf:resource="' + source.replace('&', '&amp;') + '"/>\n'
        mapString += '      <entity2 rdf:resource="' + target.replace('&', '&amp;') + '"/>\n'
        mapString += '      <relation>' + relation + '</relation>\n'
        mapString += '      <measure rdf:datatype="xsd:float">' + str(confidence) + '</measure>\n'
        mapString += '    </Cell>\n'
        mapString += '  </map>\n'
    return mapString


def serialize_mapping(alignments, ontoOne=None, ontoTwo=None, extension=None):
    return """<?xml version=\"1.0\" encoding=\"utf-8\"?>
<rdf:RDF xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
  xmlns:xsd="http://www.w3.org/2001/XMLSchema#">
<Alignment>
  <xml>yes</xml>
  <level>0</level>
  <type>??</type>
""" + \
           __get_extension_string(extension) + \
           __get_ontology_string(ontoOne, 'onto1') + \
           __get_ontology_string(ontoTwo, 'onto2') + \
           __get_mappings_string(alignments) + \
           """</Alignment>
</rdf:RDF>
"""


def serialize_mapping_to_file(file, alignments, ontoOne=None, ontoTwo=None, extension=None):
    with open(file, 'w', encoding='utf-8') as out_file:
        out_file.write(serialize_mapping(alignments, ontoOne, ontoTwo, extension))


# parser

class AlignmentHandler(object):

    def __init__(self):
        self.base = '{http://knowledgeweb.semanticweb.org/heterogeneity/alignment}'
        self.rdf = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
        self.text = ''
        self.mapping = []
        self.one_cell = ['', '', '', '']
        self.extension = {}
        self.onto1 = ''
        self.onto2 = ''
        self.onto_temp = ['', '']
        self.used_tags = set([self.base + name for name in ['entity1', 'entity2', 'relation', 'measure',
                                                            'Cell', 'map', 'Alignment', 'xml', 'level', 'type', 'onto1',
                                                            'onto2', 'Ontology', 'location',
                                                            'formalism', 'Formalism']])
        self.used_tags.add(self.rdf + 'RDF')

    def start(self, name, attrs):
        if name == self.base + 'entity1':
            self.one_cell[0] = attrs[self.rdf + 'resource']  # .encode('utf-8')
        elif name == self.base + 'entity2':
            self.one_cell[1] = attrs[self.rdf + 'resource']  # .encode('utf-8')
        elif name == self.base + 'Ontology':
            self.onto_temp[0] = attrs[self.rdf + 'about']  # .encode('utf-8')
        self.text = ''

    def end(self, name):
        if name == self.base + 'relation':
            self.one_cell[2] = self.text.strip()
        elif name == self.base + 'measure':
            self.one_cell[3] = self.text.strip()
        elif name == self.base + 'Cell':
            self.mapping.append(self.one_cell)
            self.one_cell = ['', '', '', '']
        elif name == self.base + 'location':
            self.onto_temp[1] = self.text.strip()
        elif name == self.base + 'onto1':
            if self.onto_temp[0] == '' and self.onto_temp[1] == '':
                self.onto_temp[0] = self.text.strip()
            self.onto1 = list(self.onto_temp)
        elif name == self.base + 'onto2':
            if self.onto_temp[0] == '' and self.onto_temp[1] == '':
                self.onto_temp[0] = self.text.strip()
            self.onto2 = list(self.onto_temp)
        elif name == self.base + 'measure':
            self.one_cell[3] = self.text.strip()
        elif name not in self.used_tags:
            key = name[name.index('}') + 1:]
            self.extension[key] = self.text

    def data(self, chars):
        self.text += chars

    def close(self):
        pass


def parse_mapping_from_string(s):
    handler = AlignmentHandler()
    etree.parse(BytesIO(s.encode('utf-8')), etree.XMLParser(target=handler))
    return handler.mapping, handler.onto1, handler.onto2, handler.extension


def parse_mapping_from_file(in_file):
    handler = AlignmentHandler()
    etree.parse(in_file, etree.XMLParser(target=handler))
    return handler.mapping, handler.onto1, handler.onto2, handler.extension


# build up a hashmap from mapping for faster lookup
def __get_map_from_mapping(mapping):
    map = {}
    for (source, target, relation, confidence) in mapping:
        map[(source, target)] = (source, target, relation, confidence)
    return map


def get_confusion_matrix(mapping_system, mapping_gold):
    predicted_positive = len(mapping_system)
    actual_positive = len(mapping_gold)
    true_positiv = 0
    map_system = __get_map_from_mapping(mapping_system)
    for (source, target, relation, confidence) in mapping_gold:
        if map_system.get((source, target), None) is not None:
            true_positiv += 1
            print("true_positiv {} = {}".format(source, target))
        else:
            print("should be found {} = {}".format(source, target))

    map_gold = __get_map_from_mapping(mapping_gold)
    for (source, target, relation, confidence) in mapping_system:
        if map_gold.get((source, target), None) is None:
            print("too much {} = {}".format(source, target))

    if predicted_positive == 0:
        precision = 1
    else:
        precision = true_positiv / predicted_positive

    if actual_positive == 0:
        recall = 1
    else:
        recall = true_positiv / actual_positive

    if precision == 0 and recall == 0:
        fmeasure = 0
    else:
        fmeasure = 2 * (precision * recall) / (precision + recall)

    return (true_positiv, predicted_positive, actual_positive), (precision, recall, fmeasure)


def write_gold_lexicon(mapping, src="runescape", tgt="oldschoolrunescape", test_size=20):
    lexicon = []
    counter = 0
    for map in mapping:
        if map[0].startswith("http") & map[1].startswith("http"):
            lexicon.append(("<" + map[0] + ">", "<" + map[1] + ">"))
            counter += 1

    train_lexicon, test_lexicon = lexicon[:(counter - test_size)], lexicon[-(test_size):]

    with open("../training-monolingual/lexicon/train_lexicon_{}-{}".format(src, tgt), 'wb') as train_dump:
        pickle.dump(train_lexicon, train_dump, protocol=pickle.HIGHEST_PROTOCOL)

    with open("../training-monolingual/lexicon/test_lexicon_{}-{}".format(src, tgt), 'wb') as test_dump:
        pickle.dump(test_lexicon, test_dump, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved train_lexicon, test_lexicon with sizes:", len(train_lexicon), ",", len(test_lexicon))

    return lexicon


def load_kv(src="runescape", tgt="darkscape", size_src=200, size_tgt=200, window=10):
    '''Load source and target keyed vectors of knowledge graph embeddings'''
    abs_path = os.getcwd()

    kv_filename_src = abs_path[:-6] + "/training-monolingual/kv/keyvec_src-{}_dim{}_ws{}.kv".format(src, size_src, window)
    kv_filename_tgt = abs_path[:-6] + "/training-monolingual/kv/keyvec_tgt-{}_dim{}_ws{}.kv".format(tgt, size_tgt, window)
    vecfile_src = get_tmpfile(kv_filename_src)
    vecfile_tgt = get_tmpfile(kv_filename_tgt)


    assert (os.path.isfile(kv_filename_src) & os.path.isfile(kv_filename_tgt))
    kv_src = KeyedVectors.load(vecfile_src, mmap='r')
    kv_tgt = KeyedVectors.load(vecfile_tgt, mmap='r')

    return kv_src, kv_tgt


def write_inflated_lexicon(mapping, src="runescape", tgt="darkscape", test_size=500, size_src=200,
                           size_tgt=200, window=10):
    '''
    :param mapping:
    :param src: Source knowledge graph
    :param tgt: Target knowledge graph
    :param test_size: size of test set, train size derived automatically
    :param size_src: size of source embedding
    :param size_tgt: size of target embedding
    :param window: window size of learned embeddings
    :return:
    '''
    # get vocabulary from random walk files
    source_kv, target_kv = load_kv(src=src, tgt=tgt, size_src=size_src, size_tgt=size_tgt, window=window)

    # extract properties from gold mappings
    gold_lexicon = []
    for map in mapping:
        if ("property" in map[0]) & ("property" in map[1]):
            try:
                source_kv["<" + map[0] + ">"]
                target_kv["<" + map[1] + ">"]
                gold_lexicon.append(("<" + map[0] + ">", "<" + map[1] + ">"))
            except:
                continue

    # inflate lexicon via label match
    sources = []
    targets = []
    with open("../KGs_for_gold_standard/" + src + "/enwiki-20170801-labels.ttl", 'r') as src_file:
        for line in src_file:
            start = line.find("<")
            end = line.find(">")
            resource = line[start:end + 1]
            sources.append(resource)

    with open("../KGs_for_gold_standard/" + tgt + "/enwiki-20170801-labels.ttl", 'r') as tgt_file:
        for line in tgt_file:
            start = line.find("<")
            end = line.find(">")
            resource = line[start:end + 1]
            targets.append(resource)
    sources_lower = [i.lower() for i in sources]
    targets_lower = [i.lower() for i in targets]
    match_lexicon = []

    for i, sc in enumerate(sources_lower):
        if sc.replace(src.lower(), tgt.lower()) in targets_lower:
            try:
                source_kv[sources[i]]
                target_kv[targets[targets_lower.index(sc.replace(src.lower(), tgt.lower()))]]
                match_lexicon.append((sources[i], targets[targets_lower.index(sc.replace(src.lower(), tgt.lower()))]))
            except:
                continue

    # get combined lexicon
    lexicon = gold_lexicon + match_lexicon[1:]

    train_lexicon, test_lexicon = lexicon[:-test_size], lexicon[-test_size:]

    with open("../training-monolingual/lexicon/train_lexicon_{}-{}".format(src, tgt), 'wb') as train_dump:
        pickle.dump(train_lexicon, train_dump, protocol=pickle.HIGHEST_PROTOCOL)

    with open("../training-monolingual/lexicon/test_lexicon_{}-{}".format(src, tgt), 'wb') as test_dump:
        pickle.dump(test_lexicon, test_dump, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved train_lexicon, test_lexicon with sizes:", len(train_lexicon), ",", len(test_lexicon))

    return lexicon


if __name__ == "__main__":
    mapping, onto1, onto2, extension = parse_mapping_from_file(
        '../gold_mappings/darkscape~oldschoolrunescape~evaluation.xml')
    # mapping, onto1, onto2, extension = parse_mapping_from_file('C:\\dev\\dbkwik_extraction\\extraction_docker_ubuntu\\newapproach\\e_gold_mapping_interwiki\\gold\\darkscape~oldschoolrunescape~evaluation.xml')
    # lexicon = write_gold_lexicon(mapping, src="darkscape", tgt="oldschoolrunescape")

    # write_to_file('./test.xml', [], ('Dbpedia', 'http://dbpedia.org', 'test', 'bla'))

    # keyed vectors need to be created in projections.py first
    lexicon = write_inflated_lexicon(mapping, src="darkscape", tgt="oldschoolrunescape", window=3)
