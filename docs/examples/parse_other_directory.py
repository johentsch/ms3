from ms3 import Parse
import os
p = Parse('..', file_re="mscx$", key='ms3')
mozart_path = os.path.join('~', 'mozart_piano_sonatas')
p.add_corpus(mozart_path, file_re="mscx$", corpus_name='other')
print(p)
