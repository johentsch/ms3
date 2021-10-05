from ms3 import Parse
p = Parse('..', file_re="mscx$", key='ms3')
p.add_dir('~/mozart_piano_sonatas', file_re="mscx$", key='other')
p.parse_mscx('ms3', level='c')
print(p)
