#!/usr/bin/python3

"""
NOVA.py single-nucleotide metagenomic illumina tag extractor.
"""

import yappi
from datetime import datetime

import argparse
from os import mkdir
from shutil import rmtree
from itertools import combinations

from pandas import DataFrame

import warnings
warnings.filterwarnings('ignore') # IGNORE "backend_gtk3.py:197: Warning: Source ID XXX was not found when attempting to remove it"

from lib.SeqData import SeqData, TAXRANKS
from lib.Assembler import Assembler # We want to import pandas before matplotlib (itself imported in lib.Assembler) to avoid a weird error


def main(args):

    ### Create output dir
    try:
        mkdir(args.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise
        elif not args.force_overwrite:
            print(f'\nThe directory {args.output_dir} already exists. Please remove it or use a different output name.\n', file = stderr)
            exit(1)
        else:
            rmtree(args.output_dir, ignore_errors=True)
            mkdir(args.output_dir)

    ### Write parameter log
    with open(f'{args.output_dir}/params.tsv', 'w') as outfile:
        for arg in vars(args):
            outfile.write(f'{arg}\t{getattr(args, arg)}\n')

    ### Load sequences
    seqData = SeqData(pair_delim='/')
    seqData.load_fastq('/home/fer/Projects/smiTE/testsNOVA/mock3/samples/Oral3.mock3.S_0.perfect.InSilicoSeq_R1.fastq', 'S_0')
    seqData.load_fastq('/home/fer/Projects/smiTE/testsNOVA/mock3/samples/Oral3.mock3.S_0.perfect.InSilicoSeq_R2.fastq', 'S_0', rc = True)
##    seqData.load_fastq('/home/fer/Projects/smiTE/testsNOVA/Freshwaters/mock1/samples/Oral3.mock3.S_0.perfect.InSilicoSeq_R1.fastq', 'S_0')
##    seqData.load_fastq('/home/fer/Projects/smiTE/testsNOVA/Freshwaters/mock1/samples/Oral3.mock3.S_0.perfect.InSilicoSeq_R2.fastq', 'S_0', rc = True)
    if args.tax_level:
        #seqData.classify_mothur(args.output_dir, '/home/fer/DB/silva.nr.v132/silva.nr_v132.align', args.tax_file, args.processors)
        #seqData.classify_mothur(args.output_dir, '/home/fer/Projects/smiTE/DB/sub10/silva.sub10.subsample.1.align', args.tax_file, args.processors)
        seqData.classify_mothur(args.output_dir, '/home/fer/Projects/smiTE/DB/sub100/silva.sub100.subsample.1.align', args.tax_file, args.processors)
        seqData.correct_tax_paired(args.tax_level)
            
##    seqData = SeqData()
##    seqData.load_fasta(args.align_file)
##    seqData.load_tax_mothur(args.align_report, args.tax_file)
##    seqData.expand_sequences_mothur(args.names)
##    seqData.load_samples_mothur(args.groups)
##    if args.tax_level:
##      seqData.correct_tax_paired(args.tax_level)


##    seqData = SeqData(pair_delim='/')
##    seqData.load_fasta(args.align_file, 'S_0')
##    seqData.load_tax_mothur(args.align_report, args.tax_file)
##    if args.tax_level:
##      seqData.correct_tax_paired(args.tax_level)


    print(f'\nLoaded {len(seqData.sequences)} sequences\n')

    ### Reconstruct full sequences for each taxon and sample
    taxa = sorted({tax['tax'][args.tax_level] for tax in seqData.taxonomy.values()}) if args.tax_level else ['All taxa']
    samples = sorted(set(seqData.samples.values()))
    ASVs = {sample: {} for sample in samples}
    residuals = {}
    seqTaxa = {}

    for taxon in taxa:
        #if taxon != 'Lactobacillales': continue
        #if taxon != 'Neisseriaceae': continue
        if taxon == 'Unclassified': continue
        
        residuals[taxon] = {}
        
        mkdir(f'{args.output_dir}/{taxon}')
        taxSeqData = seqData.subset_tax(args.tax_level, taxon) if taxon != 'All taxa' else seqData

        results = [run_assembler(taxSeqData.subset_samples(sample), sample, taxon, args) for sample in samples]
        
        for sample, (seqAbunds, residual) in zip(samples, results):
            residuals[taxon][sample] = residual
            for seq, abund in seqAbunds.items():
                if seq in seqTaxa and seqTaxa[seq] != taxon:
                    # We should not be retrieving the exact same sequence when working with two different taxa, but it may happen
                    #  In that case, warn and just add the abundances to the existing ones
                    # seqTaxa[seq] != taxon because we can and will retrieve the exact same sequence with the same taxa BUT IN TWO DIFFERENT SAMPLES
                    #  In that case we don't execute this part, but in the else part we assert that this sequence is not already in the sample, which should never happen if our alignment was properly corrected
                    print(f'WARNING! Sequence "{seq}" was recovered from taxa {taxon} and {seqTaxa[seq]}')
                    if seq in ASVs[sample]:
                        ASVs[sample][seq] += abund
                    else:
                        ASVs[sample][seq] = abund                        
                else:
                    assert seq not in ASVs[sample]
                    assert seq not in seqTaxa
                    ASVs[sample][seq] = abund
                    seqTaxa[seq] = taxon
                    
        # Sometimes we get two different ASVs in two samples, one of which is a subsequence of the other. Check that and combine them if required
        taxSeqs = [seq for seq, taxon in seqTaxa.items()] # this will complain all the time but honestly we want to notice these issues.
        subSeqs = {} # {sub: full}
        
        def check_subseq(seq1, seq2):
            if seq1 in seq2:
                if len(seq2) - len(seq1) >= 3:
                    msg = f'WARNING! Sequence "{seq1}" is a subset of "{seq2}" but it\'s too small. Ignoring merge!'
                    msg += f'\n{seqTaxa[seq1]} {seqTaxa[seq2]}'
                    print(msg)
                else:
                    subSeqs[seq1] = seq2
            
        for seq1, seq2 in combinations(taxSeqs, 2):
            check_subseq(seq1, seq2)
            check_subseq(seq2, seq1)

        # Assert there are no subset chains
        for seq1, seq2 in subSeqs.items():
            assert seq2 not in subSeqs

        # Merge subsequences into larger sequences
        for seq1, seq2 in subSeqs.items():
            del seqTaxa[seq1]
            for sample in samples:
                if seq1 in ASVs[sample]:
                    if seq2 in ASVs[sample]:
                        ASVs[sample][seq2] += ASVs[sample][seq1]
                    else:
                        ASVs[sample][seq2] = ASVs[sample][seq1]
                    del ASVs[sample][seq1]

    ### Generate ASV matrix with nice names
    ASVmatrix = DataFrame.from_dict(ASVs).fillna(0).transpose()
    ASVnames = {}
    abundSortedASVs = ASVmatrix.sum(0).sort_values(ascending=False)
    ASVmatrix = ASVmatrix.loc[:,abundSortedASVs.index]
    for i, v in enumerate(abundSortedASVs.items()):
        seq, abund = v
        inSamples = '|'.join(sorted([sample for sample in ASVs if seq in ASVs[sample]]))
        if 'Unrecovered' not in seq:
            shortName = f'ASV{i+1}_{seqTaxa[seq]}'
        else:
            shortName = seq
        longName = f'{shortName} size={int(abund)} samples={inSamples}'
        ASVnames[seq] = {'short': shortName, 'long': longName}
        
    ASVmatrix.columns = [ASVnames[seq]['short'] for seq in ASVmatrix.columns]
    ASVmatrix.astype(int).to_csv(args.output_dir + '/' + 'ASVs.counts.matrix', sep='\t')
    ASVmatrix.div(ASVmatrix.sum(1), axis=0).to_csv(args.output_dir + '/' + 'ASVs.relAbund.matrix', sep='\t')

    ### Write fasta output.
    with open(args.output_dir + '/' + 'ASVs.fasta', 'w') as outfile:
        for seq in abundSortedASVs.keys():
            if 'Unrecovered' not in seq:
                longName = ASVnames[seq]['long']
                outfile.write(f'>{longName}\n{seq}\n')


def run_assembler(seqData, sample, taxon, args): # Need an independent function since a lambda can't be pickled.
    #seqData.to_fasta(f'{args.output_dir}/{taxon}/{sample}.{taxon}.fasta')
    assembler = Assembler(seqData, args.pe_support_threshold, args.processors, sample, taxon, args.output_dir)
    return assembler.run(args.ksize)



def parse_args():
    parser = argparse.ArgumentParser(description='Resolve Metagenomic Sequence Variants')
    parser.add_argument('-a', '--align-file', type = str, required = True,
                        help='Align file')
    parser.add_argument('-r', '--align-report', type = str, required = True,
                        help='Align report')
    parser.add_argument('-n', '--names', type = str, required = True,
                        help='Mothur names file')
    parser.add_argument('-g', '--groups', type = str, required = True,
                        help='Mothur groups file')
    parser.add_argument('-s', '--tax-file', type = str, default= '/home/fer/DB/silva.nr.v132/silva.nr_v132.tax',
                        help='SILVA taxonomy file matching the reference using for aligning the sequences')
    parser.add_argument('-t', '--tax-level', type = str,
                        help='Taxononomic level to resolve')
    parser.add_argument('-o', '--output-dir', type = str, default = 'output',
                        help = 'Output directory')
    parser.add_argument('-k', '--ksize', type = int, default = 96,
                        help = 'kmer size')
    parser.add_argument('-z', '--pe_support_threshold', type = float, default = 1,
                        help = 'Support threshold for discarding paths using the paired-end method')
    parser.add_argument('-p', '--processors', type = int, default = 1,
                        help = 'Number of processors to use (maximum is one processor per sample')
    parser.add_argument('--force-overwrite', action='store_true',
                        help = 'Force overwrite if the output directory already exists')
    parser.add_argument('--profile', action='store_true',
                        help = 'Profile execution using yappi')
    parser.add_argument('--verbose', action='store_true',
                        help = 'Output detailed info')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    if not args.profile:
        main(args)
    else:
        yappi.start()
        try:
            main(args)
        finally:
            func_stats = yappi.get_func_stats()
            func_stats.save('callgrind.out.' + datetime.now().isoformat(), 'CALLGRIND')
            yappi.stop()
            yappi.clear_stats()
