#!/usr/bin/env python
'''
 (Note that they'll only work with files less than
 4,294,967,296 bytes long.. If your files are larger than that
 substitute 'Q' for 'L' in the struct formats.)


 Usage: "getline.py <datafile<indexfile<num>"

 Prints line num from datafile using indexfile.
'''
import struct
import sys

#fmt = 'L' ## unsigned long
fmt = 'Q' ## unsigned long long

fmt_size = struct.calcsize(fmt)


def F(n, fn):
    '''
    Return the byte offset of line n from index file fn.
    '''
    f = open(fn)

    try:
        f.seek(n * fmt_size)
        data = f.read(fmt_size)
    finally:
        f.close()

    return struct.unpack(fmt, data)[0]


def getline(n, data_file, index_file):
    '''
    Return line n from data file using index file.
    '''
    n = F(n, index_file)
    f = open(data_file)

    try:
        f.seek(n)
        data = f.readline()
    finally:
        f.close()

    return data


if __name__ == '__main__':
    dfn, ifn, lineno = sys.argv[-3:]
    n = int(lineno)
    print getline(n, dfn, ifn)
