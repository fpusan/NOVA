from sys import stderr
from os import devnull
from copy import deepcopy
from collections import defaultdict
from math import ceil
from subprocess import call

import numpy as np
from scipy.stats import entropy

TAXRANKS = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

class Alignment():
                                         
    def __init__(self, sequences, names = list()):
        if isinstance(sequences, np.ndarray):
            self.array = sequences
        else:
            self.array = np.array([list(seq) for seq in sequences], dtype = np.dtype('U1'))

        self.shape = self.array.shape
        self.names = names
        self.indices = {name: i for i, name in enumerate(names)} # {sequence name: index in the array}
        self.idxToName = {i:n for n, i in self.indices.items()}  # {index in the array: sequence name}
        self.references = dict()                                 # {sequence name: hit in ref db}
        self.taxonomy = dict()                                   # {sequence name: {rank: taxon}}
        self.samples = dict()                                    # {sequence name: sample}
        self.pairings = dict()
        pairs = defaultdict(list)
        for name in self.names:
            pair = name.split('_pair_')[0]
            pairs[pair].append(name)
                            
        for pair, names in pairs.items():
            if len(names) == 1:
                self.pairings[names[0]] = None
            elif len(names) == 2:
                self.pairings[names[0]] = names[1]
                self.pairings[names[1]] = names[0]
            else:
                raise Exception('Wtf')
        self.coverage, self.entropy, self.consensus = self.profile_alignment()
        self.medianReadLength, self.medianPairSep = self.calculate_median_lengths()

    def __len__(self):
        return self.array.shape[0]


    def __repr__(self):
        return self.array.__repr__()


    def __getitem__(self, key):
        
        def parse_row_key(key):
            if type(key) is slice:
                start, stop, step = key.start, key.stop, key.step
                start = 1 if not start else start
                stop = self.shape[0] + 1 if not stop else stop
                step = 1 if not step else step
                idxs = [i-1 for i in range(start, stop, step)]
                for idx in idxs:
                    if idx not in self.idxToName:
                        raise IndexError('Index {} is out of bounds for axis 0 with size {}'.format(idx, self.shape[0]))
                names = [self.idxToName[idx] for idx in idxs] if self.idxToName else []
            elif type(key) is int:
                if key not in self.idxToName:
                    raise IndexError('Index {} is out of bounds for axis 0 with size {}'.format(key, self.shape[0]))
                idxs = [key]
                names = [self.idxToName[key]] if self.idxToName else []
            elif type(key) is str:
                if not self.indices:
                    raise TypeError('String indexing is not supported in this alignment')
                idxs = [self.indices[key]]
                names = [key]
            elif type(key) is tuple or type(key) is list:
                idxs = []
                names = []
                for k2 in key:
                    if type(k2) is int or type(k2) is str:
                        idxs2, names2 = parse_row_key(k2)
                        assert len(idxs2) == 1 and len(names2) == 1
                        idxs.append(idxs2[0])
                        names.append(names2[0])
                    else:
                        raise TypeError('Only str or int are valid row keys for an alignment')
            else:
                raise TypeError('Only str or int are valid row keys for an alignment')

            return idxs, names
                
        if type(key) is tuple: # First dimension is rows, second is columns
            assert len(key) > 0
            idxs, names = parse_row_key(key[0])
            if len(key) == 1:
                columns = None
            elif len(key) == 2:
                columns = key[1]
            else:
                raise IndexError('too many indices for alignment')
        else:
            idxs, names = parse_row_key(key)
            columns = None

        if not columns:
            array = self.array[idxs]
        else:
            if type(columns) is list and type(columns[0]) is bool:
                array = self.array[idxs][:,columns] # Hack due to https://github.com/numpy/numpy/issues/13255
            else:
                array = self.array[idxs, columns]

        references = {name: self.references[name] for name in names} if self.references else {}
        taxonomy = {name: self.taxonomy[name] for name in names} if self.taxonomy else {}
        samples = {name: self.samples[name] for name in names} if self.samples else {}

        obj = Alignment(deepcopy(array), names)
        obj.references = references
        obj.taxonomy = taxonomy
        obj.samples = samples
        return obj


    @staticmethod
    def from_fasta(fasta, check_Ns = False):
        sequences = []
        names = []
        with open(fasta) as infile:
            rawSeqs = infile.read().strip('>\n').split('>')
            for seq in rawSeqs:
                name, seq = seq.strip().split('\n', 1)
                seq = seq.replace('\n','').upper() # uppercase!
                # Substitute leading/trailing Ns by dots.
                if check_Ns:
                    seq = list(seq)
                    for i in range(len(seq)):
                        if seq[i] not in ('.', '-', 'N'):
                            break
                        if seq[i] == ('N', '-'):
                            seq[i] = '.'
                    for i in range(len(seq)):
                        if seq[-i] not in ('.', '-', 'N'):
                            break
                        if seq[-i] in ('N', '-'):
                            seq[-i] = '.'
                    seq = ''.join(seq)
                sequences.append(seq)
                names.append(name)
            # Allow different sequence lenghts but complain
            maxLength = max([len(seq) for seq in sequences])
            ok = True
            for i, seq in enumerate(sequences):
                if len(seq) < maxLength:
                    ok = False
                    sequences[i] = '{}{}'.format( seq, '-' * (maxLength - len(seq)) ) # gap instead of dot to avoid empty lists when calculating entropies
            if not ok:
                print('WARNING: Sequences in the "{}" file did not have the same length. This should not happen unless you are using refined data from a previous run of the program'.format(fasta), file = stderr)
                    
        return Alignment(sequences, names)


    def to_fasta(self, output_name, dots_to_gaps = False, append = False):
        names = [i+1 for i in range(len(self))] if not self.names else self.names
        mode = 'a' if append else 'w'
        with open(output_name, mode) as outfile:
            for name, row in zip(names, self.array):
                seq = ''.join(row) if not dots_to_gaps else ''.join(row).replace('.','-')
                outfile.write('>{}\n{}\n'.format(name, seq))


    def to_seqdict(self, degap = False):
        seqdict = {}
        names = [i+1 for i in range(len(self))] if not self.names else self.names
        for name, row in zip(names, self.array):
            seq = ''.join(row) if not degap else ''.join(row).replace('-','.').replace('.','')
            seqdict[name] = seq
        return seqdict


    def get_unique_names(self):
        """
        Get a dictionary with the names for representative sequences as keys, and lists with the names of represented sequences as values.
        """
        assert len(self.names) == self.array.shape[0]
        uniques = {}
        namedict = {}
        for name, seq in zip(self.names, self.array):
            seq = ''.join(seq)
            if seq not in uniques:
                uniques[seq] = name
                namedict[name] = [name]
            else:
                namedict[uniques[seq]].append(name)
        return namedict
            

    def expand_sequences_mothur(self, names_file):
        namedict = {line.split('\t')[0]: line.split('\t')[1].strip().split(',') for line in open(names_file)}
        return self.expand_sequences(namedict)


    def expand_sequences(self, namedict):
        assert len(self.names) == self.array.shape[0]
##        if self.samples:
##            print('WARNING: sample information will not be carried to the new object')        

        sequences = []
        names = []
        references = {}
        taxonomy = {}
        samples = {}
        
        for name, row in zip(self.names, self.array):
            for exp in namedict[name]:
                names.append(exp)
                sequences.append(row)
                if self.references:
                    references[exp] = self.references[name]
                if self.taxonomy:
                    taxonomy[exp] = self.taxonomy[name]
                    
        obj = Alignment(sequences, names)
        obj.references = references
        obj.taxonomy = taxonomy
        return obj
      

    def load_samples_mothur(self, groups_file):
        self.samples = dict([line.strip().split('\t') for line in open(groups_file)])
    

    def load_tax_mothur(self, report_file, ref_tax_file):

        MIN_SEARCH_SCORE = 0
        MIN_IDENTITY = 0
        
        self.references = {}
        with open(report_file) as infile:
            infile.readline() # Burn header
            for line in infile:
                if '-nan' in line:
                    print(line)
                    line = line.replace('-nan', '')
                line = line.strip().split('\t')
                name, ref, searchscore, ident = line[0], line[2], float(line[5]), float(line[-1])
                self.references[name] = {'ref': ref, 'searchscore': searchscore, 'ident': ident}                    
                
        refTax = dict([line.strip().split('\t') for line in open(ref_tax_file)])
        self.taxonomy = {}
        for name, refdict in self.references.items():
            if refdict['searchscore'] >= MIN_SEARCH_SCORE and refdict['ident'] >= MIN_IDENTITY:
                tax = refTax[refdict['ref']].strip(';').split(';')
                lastTax = tax[-1]
                while len(tax) < len(TAXRANKS):
                    tax.append('Unclassified {}'.format(lastTax))
                self.taxonomy[name] = dict(zip(TAXRANKS, tax))
            else:
                self.taxonomy[name] = {rank: 'Unclassified' for rank in TAXRANKS}
                

    def subset_tax(self, level, taxon, strip = False):
        goodNames = [name for name, tax in self.taxonomy.items() if tax[level] == taxon]
        res = self[goodNames,]
        if strip:
            res = res.strip_uncovered_positions()
        return res


    def subset_samples(self, sample, strip = False):
        goodNames = [name for name, sam in self.samples.items() if sam == sample]
        res = self[goodNames,]
        if strip:
            res = res.strip_uncovered_positions()
        return res
                        

    def add_dots(self):
        for i in range(self.array.shape[0]):
            for j in range(self.array.shape[1]):
                if self.array[i,j] not in ('-', '.'):
                    break
                else:
                    self.array[i,j] = '.'
            for j in range(self.array.shape[1]):
                if self.array[i,-j-1] not in ('-', '.'):
                    break
                else:
                    self.array[i,-j-1] = '.'


    def strip_uncovered_positions(self):
        """
        Remove leading trailing positions with zero coverage
        """
        leftLim = rightLim = 0
        for i in range(self.array.shape[1]):
            symbols = np.unique(self.array[:,i])
            if len(symbols) == 1 and symbols[0] == '.':
                continue
            else:
                leftLim = i
                break
        for i in reversed(range(self.array.shape[1])):
            symbols = np.unique(self.array[:i])
            if len(symbols) == 1 and symbols[0] == '.':
                continue
            else:
                rightLim = i+1
                break
        assert leftLim <= rightLim
        if not leftLim and not rightLim:
            return self
        else:
            return self[:,leftLim:rightLim]


    def filter_gaps(self):
        """
        Remove full gap columns.
        """
        good = [not (consensus in ('.', '-') and entropy==0) for consensus, entropy in zip(self.consensus, self.entropy)]
        return self[:,good]


    def calculate_median_lengths(self):
        """
        Calculate median read length and pair separation. If there is no pairing info, pair separation will be 0 (don't trust this to check whether there are pairs, rather look at the content of self.pairings).
        If reads overlap pair separation will be negative.
        """
        startPos = {}
        endPos = {}
        emptySeqs = set()
        for i in range(self.array.shape[0]):
            for j in range(self.array.shape[1]):
                if self.array[i,j] != '.':
                    startPos[i] = j
                    break
            else:
                emptySeqs.add(i)
                continue
            for j in range(self.array.shape[1]-1, -1, -1):
                if self.array[i,j] != '.':
                    endPos[i] = j
                    break
        
        lengths = [endPos[i] - startPos[i]+1 for i in startPos] # Note that we are including gaps here
        if self.pairings:
            pairSep = []
            for p1, p2 in self.pairings.items():
                if p1 and p2:
                    if self.indices[p1] not in emptySeqs and self.indices[p2] not in emptySeqs:
                        if startPos[self.indices[p1]] < startPos[self.indices[p2]]:
                            pairSep.append(startPos[self.indices[p2]] - endPos[self.indices[p1]] - 1)
            pairSep = np.median(pairSep)
        else:
            pairSep = 0
        return np.median(lengths), pairSep


    def profile_alignment(self):
        """
        Generate coverage, entropy and consensus profiles for the aligment.
        """
        cov = []
        ent = []
        cons = []
        for j in range(self.array.shape[1]):
            freqs = {symbol: freq for symbol, freq in zip(*np.unique(self.array[:,j], return_counts = True)) if symbol != '.'}
            if not freqs: # When having a poor alignment to a taxon we can get positions with coverage 0
                cov.append(0)
                ent.append(0)
                cons.append('.')
            else:
                mostCommonSymbol = max(freqs, key=lambda x: freqs[x])
                S = entropy(list(freqs.values()))
                cov.append(sum(freqs.values()))
                ent.append(S)
                cons.append(mostCommonSymbol)

        return cov, ent, cons        

