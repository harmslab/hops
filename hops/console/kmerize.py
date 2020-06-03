#!/usr/bin/env python3
__description__ = \
"""
Take a fasta file and generate all kmers from that file.
"""
__author__ = "Michael J. Harms"
__date__ = "2017-06-30"
__usage__ = "proteomer.py fasta_file kmer_size"

import sys, random, argparse, os

def create_kmers(seq,kmer_size):
    """
    Create a set of kmers from a sequence.
    """

    return [seq[i:(i+kmer_size)] for i in range(len(seq)-kmer_size+1)]

def parse_proteome(fasta_file,kmer_size=12,out_base="kmers",seq_per_file=50000,num_to_write=1000000):
    """
    Read a uniprot fasta file of protein sequences and break into a set of
    kmers.
    """

    all_kmers = {}
    seq_name = None
    current_sequence = []

    # Parse fasta file, splitting into kmers as we go
    with open(fasta_file) as infile:
        for l in infile:

            if l.startswith(">"):
                if seq_name is not None:

                    sequence = "".join(current_sequence)
                    kmer_list = create_kmers(sequence,kmer_size)

                    for k in kmer_list:
                        try:
                            all_kmers[k].append(seq_name)
                        except KeyError:
                            all_kmers[k] = [seq_name]

                current_sequence = []
                seq_name = l[1:].strip()
            else:
                if seq_name is None or l.strip() == "":
                    continue
                current_sequence.append(l.strip())

    if seq_name is not None:

        sequence = "".join(current_sequence)
        kmer_list = create_kmers(sequence,kmer_size)

        for k in kmer_list:
            try:
                all_kmers[k].append(seq_name)
            except KeyError:
                all_kmers[k] = [seq_name]

    # Sort kmers
    to_sort = [(len(all_kmers[k]),k) for k in all_kmers.keys()]
    to_sort.sort(reverse=True)

    # kmers 
    kmers = [k[1] for k in to_sort]

    if len(kmers) > num_to_write:
        kmers = kmers[:num_to_write]
    else:

        # If there are more single kmers than the total we want to get, grab a
        # random selection of them.
        single_kmers = [k[1] for k in to_sort if k[0] == 1]
        if num_to_write - len(kmers) > 0:
            to_grab = num_to_write - len(kmers)
            random.shuffle(single_kmers)
            kmers.extend(single_kmers[:to_grab])

    out = []
    counter = 0
    for k in kmers:

        # make sure kmer has only amino acids in it
        score = sum([1 for l in k if l not in "ACDEFGHIKLMNPQRSTVWY"])
        if score > 0:
            continue

        ids = ",".join(all_kmers[k])
        out.append("{} {:5d} {}\n".format(k,len(all_kmers[k]),ids))

        if counter != 0 and counter % seq_per_file == 0:

            out_file = "{}_{}.kmers".format(out_base,counter)
            print(counter,len(kmers))
            sys.stdout.flush()

            f = open(out_file,'w')
            f.write("".join(out))
            f.close()

            out = []

        counter += 1


    out_file = "{}_{}.kmers".format(out_base,counter)

    f = open(out_file,'w')
    f.write("".join(out))
    f.close()


def main(argv=None):
    """
    If run from the command line.
    """

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description=__description__)

    # Positionals
    parser.add_argument("fasta_file",help="fasta file to be turned into kmers")

    # Options
    parser.add_argument("-o","--outbase",help="base name for output files",action="store",type=str,default=None)
    parser.add_argument("-k","--kmersize",help="kmer size",action="store",type=int,default=12)
    parser.add_argument("-s","--seqperfile",help="number of sequences per output file",action="store",
                        type=int,default=50000)
    parser.add_argument("-n","--numkmers",
                        help="Number of kmers to make, starting from most to least common.  If -1, make all possible.",
                        type=int,default=1000000)

    args = parser.parse_args(argv)

    if args.outbase is None:
        out_base = args.fasta_file
    else:
        out_base = args.outbase

    parse_proteome(args.fasta_file,kmer_size=args.kmersize,out_base=out_base,
                   seq_per_file=args.seqperfile,num_to_write=args.numkmers)

if __name__ == "__main__":
    main()
