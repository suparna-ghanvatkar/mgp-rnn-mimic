'''
python script to read filenames from wavefiles.txt and mv them to waves folder
'''
import os

src = '/data/suparna/MGP_data/waveforms/'
dest = '/data/suparna/MGP_data/waves/'
fhandle = open('wavefile.txt')
for line in fhandle:
    line = line.strip('\n')
    src_path = src+line
    dest_path = dest+line
    os.rename(src_path, dest_path)
