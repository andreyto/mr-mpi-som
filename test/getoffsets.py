#!/usr/bin/env python
##
## Write the byte offset of each line.
##
import fileinput
import struct
import sys

def f(n): 
    #return struct.pack('L', n)
    return struct.pack('Q', n) # unsigned long long

def main():
    total = 0

    # Main processing..
    for n, line in enumerate(fileinput.input()):
        sys.stdout.write(f(total))
        total += len(line)

    # Status output.
    if not n % 1000:
        print sys.stderr, '%i lines processed' % n
    print sys.stderr, '%i lines processed' % (n + 1)
    
if __name__ == '__main__':
    main()
