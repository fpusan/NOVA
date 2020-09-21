from collections import defaultdict
from subprocess import run, DEVNULL
from copy import deepcopy

TAXRANKS = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

class SeqData():

    def __init__(self, sequences = {}, samples = {}, taxonomy = {}, pairings = {}, pair_delim = '_pair_'):
        self.sequences = sequences
        self.samples = samples
        self.taxonomy = taxonomy
        self.pairings = pairings
        self.pair_delim = pair_delim
        

    def get_pairings(self):
        pairs = defaultdict(list)
        pairings = {}
        for header in self.sequences:
            pair = header.split(self.pair_delim)[0]
            pairs[pair].append(header)
        for pair, names in pairs.items():
            if len(names) == 1:
                pairings[names[0]] = None
            elif len(names) == 2:
                pairings[names[0]] = names[1]
                pairings[names[1]] = names[0]
            else:
                raise Exception('Wtf')
        self.pairings = pairings


    def load_fasta(self, fasta, sample = None, rc = False):
        for seq in open(fasta).read().strip().lstrip('>').split('>'):
            name, seq = seq.split('\n',1)
            name = name.split('\t')[0]
            assert name not in self.sequences
            seq = seq.replace('\n','').replace('.','').replace('-','').replace('N','')
            if rc:
                seq = self.reverse_complement(seq)
            self.sequences[name] = seq
            if sample:
                self.samples[name] = sample
        self.get_pairings()


    def load_fastq(self, fastq, sample = None, rc = False):
        with open(fastq) as infile:
            while True:
                name = infile.readline().strip()[1:]
                seq = infile.readline().strip().replace('N','')
                infile.readline()
                infile.readline()
                if not name:
                    break
                if not seq:
                    continue
                if rc:
                    seq = self.reverse_complement(seq)
                self.sequences[name] = seq
                if sample:
                    self.samples[name] = sample           
        self.get_pairings()


    @staticmethod
    def reverse_complement(seq):
            complementarity_matrix = {'A':'T', 'C':'G', 'T':'A', 'G':'C', 'N':'N', 
                            'W':'W', 'S':'S', 'R':'Y', 'Y':'R', 'M':'K', 
                            'K':'M', 'B':'V', 'V':'B', 'D':'H', 'H':'D',
                            '-':'-', '.':'.'}
            return ''.join([complementarity_matrix[b] for b in seq[::-1]])
        

    def to_fasta(self, fasta):
        with open(fasta, 'w') as outfile:
            for name, seq in self.sequences.items():
                outfile.write(f'>{name}\n{seq}\n')


    def classify_mothur(self, pre_dir, ref_align_file, ref_tax_file, processors):
        print(f'\nAligning sequences with mothur')
        self.to_fasta(f'{pre_dir}/seqs.fasta')
        run([f'mothur "#align.seqs(fasta={pre_dir}/seqs.fasta, reference={ref_align_file}, processors={processors}, outputdir={pre_dir})"'], shell=True, stdout=DEVNULL)
        self.sequences = {}
        self.load_fasta(f'{pre_dir}/seqs.align')
        self.load_tax_mothur(f'{pre_dir}/seqs.align.report', ref_tax_file)
        

    def expand_sequences_mothur(self, names_file): # samples information is lost
        namedict = {line.split('\t')[0]: line.split('\t')[1].strip().split(',') for line in open(names_file)}
        sequences = {}
        taxonomy = {}
        for rep, names in namedict.items():
            for name in names:
                sequences[name] = self.sequences[rep]
                if rep in self.taxonomy:
                    taxonomy[name] = self.taxonomy[rep]
        res = SeqData(sequences = sequences, taxonomy = taxonomy, pair_delim = self.pair_delim)
        res.get_pairings()
        return res
        

    def load_samples_mothur(self, groups_file):
        self.samples = dict([line.strip().split('\t') for line in open(groups_file)])
    

    def load_tax_mothur(self, report_file, ref_tax_file):

        MIN_SEARCH_SCORE = 0
        MIN_IDENTITY = 0

        refTax = dict([line.strip().split('\t') for line in open(ref_tax_file)])
        
        with open(report_file) as infile:
            infile.readline() # Burn header
            for line in infile:
                if '-nan' in line:
                    print(line)
                    line = line.replace('-nan', '')
                line = line.strip().split('\t')
                name, ref, searchscore, qstart, qend, rstart, rend, riden = line[0], line[2], line[5], line[7], line[8], line[9], line[10], line[-1]
                qstart, qend = int(qstart), int(qend)
                alen = qend - qstart
                searchscore, riden = float(searchscore), float(riden)
                
                if searchscore >= MIN_SEARCH_SCORE and riden >= MIN_IDENTITY:
                    tax = refTax[ref].strip(';').split(';')
                    lastTax = tax[-1]
                    while len(tax) < len(TAXRANKS):
                        tax.append('Unclassified {}'.format(lastTax))
                    tax = dict(zip(TAXRANKS, tax))
                else:
                    tax = {rank: 'Unclassified' for rank in TAXRANKS}
                
                self.taxonomy[name] = {'ref': ref, 'rstart': rstart, 'rend': rend, 'alen': alen, 'riden': riden, 'searchscore': searchscore, 'score': round(alen*riden/100), 'tax': tax}

##        # Unify references from paired end data.
##        for pair1, pair2 in self.pairings.items(): # we do the same thing twice, really, but whatever.
##            if pair2:
##                searchscore1, id1 = self.taxonomy[pair1]['searchscore'], self.taxonomy[pair1]['riden']
##                searchscore2, id2 = self.taxonomy[pair2]['searchscore'], self.taxonomy[pair2]['riden']
##                if id1 >= id2 and searchscore1 >= MIN_SEARCH_SCORE and id1 >= MIN_IDENTITY:
##                    self.taxonomy[pair2] = self.taxonomy[pair1]
##                elif id2 > id1 and searchscore2 > MIN_SEARCH_SCORE and id2 >= MIN_IDENTITY:
##                    self.taxonomy[pair1] = self.taxonomy[pair2]
                


    def correct_tax_paired(self, tax_correction_level = 'family'):
        pairs = defaultdict(list)
        for name in self.taxonomy:
            pair = name.split(self.pair_delim)[0]
            pairs[pair].append(name)

        pairTax = defaultdict(lambda: defaultdict(int))
        for reads in pairs.values():
            if len(reads) == 1:
                continue
            elif len(reads) == 2:
                pair1, pair2 = reads
                tax1 = self.taxonomy[pair1]['tax'][tax_correction_level]
                tax2 = self.taxonomy[pair2]['tax'][tax_correction_level]
                pairTax[tax1][tax2] += 1
                pairTax[tax2][tax1] += 1

            else:
                raise Exception('wtf')

        # We now consider that any taxon for which there was no pair such that both members were classified to that taxon is an artifact.
        artifacts = {taxon for taxon in pairTax if taxon not in pairTax[taxon] or 'nclassified' in taxon or pairTax[taxon][taxon] / sum(pairTax[taxon].values()) < 0.1}

        def blind_correct(tax):
            pairTax2 = deepcopy(pairTax) # If I don't deepcopy here the pairTax dict gets modified somehow.
            c = 0
            while True: # potentially infinite loop
                c+=1
                if c==10:
                    newTax = tax
                    break
                bestOptions = list(sorted(pairTax2[tax], key = lambda x: pairTax2[tax1][x], reverse = True))
                if not bestOptions:
                    newTax = tax
                    break
                bestOptionsNoArtifact = [t for t in bestOptions if t not in artifacts]
                if bestOptionsNoArtifact:
                    newTax = bestOptionsNoArtifact[0]
                    break
                else:
                    tax = bestOptions[0] # search for a noartifact taxonomy starting from the best option we had in this tax.
                    
            res = {rank: 'CORRECTED' for rank in TAXRANKS}
            res[tax_correction_level] = newTax
            return res

                    
        for reads in pairs.values():
            if len(reads) == 1:
                single = reads[0]
                tax = self.taxonomy[single]['tax'][tax_correction_level]
                if tax in artifacts:
                    self.taxonomy[single]['tax'] = blind_correct(tax)
            else: # len(reads) == 2:
                pair1, pair2 = reads
                tax1 = self.taxonomy[pair1]['tax'][tax_correction_level]
                tax2 = self.taxonomy[pair2]['tax'][tax_correction_level]                  

                if tax1 != tax2 or tax1 in artifacts or tax2 in artifacts:
                    if tax1 not in artifacts and tax2 in artifacts:
                        goodTax =  self.taxonomy[pair1]['tax']        
                    elif tax1 in artifacts and tax2 not in artifacts:
                        goodTax =  self.taxonomy[pair2]['tax']
                    else: # Both are artifacts or both are non artifacts
                        if self.taxonomy[pair1]['score'] > self.taxonomy[pair2]['score']:
                            if tax1 in artifacts: # both artifacts
                                goodTax = blind_correct(tax1)
                            else:
                                goodTax =  self.taxonomy[pair1]['tax']
                        elif self.taxonomy[pair1]['score'] < self.taxonomy[pair2]['score']:
                            if tax1 in artifacts: # both artifacts
                                goodTax = blind_correct(tax2)
                            else:
                                goodTax =  self.taxonomy[pair2]['tax']
                        else: # same score!
                            if sum(pairTax[tax1].values()) > sum(pairTax[tax2].values()):
                                if tax1 in artifacts: # both artifacts
                                    goodTax = blind_correct(tax1)
                                else:
                                    goodTax =  self.taxonomy[pair1]['tax']
                            elif sum(pairTax[tax1].values()) < sum(pairTax[tax2].values()):
                                if tax1 in artifacts: # both artifacts
                                    goodTax = blind_correct(tax2)
                                else:
                                    goodTax =  self.taxonomy[pair2]['tax']
                            else: # same score and same abundance!
                                print('This message being printed means that we can\'t obtain meaningful knowledge nor make truly informed decisions. Embracing nihilism...')
                    self.taxonomy[pair1]['tax'] =  goodTax
                    self.taxonomy[pair2]['tax'] =  goodTax
                
                            
        for pair1, pair2 in self.pairings.items():
            if pair2:
                tax1 = self.taxonomy[pair1]['tax'][tax_correction_level]
                tax2 = self.taxonomy[pair2]['tax'][tax_correction_level]
                assert tax1==tax2


                    

    def subset_tax(self, level, taxon):
        return self.subset_seqs([name for name in self.sequences if self.taxonomy[name]['tax'][level] == taxon])


    def subset_samples(self, sample):
        return self.subset_seqs([name for name in self.sequences if self.samples[name] == sample])


    def subset_seqs(self, names):
        sequences = {}
        samples = {}
        taxonomy = {}
        hasTax = True
        hasSamples = True
        for name in names:
            sequences[name] = self.sequences[name]
            if hasTax:
                if name in self.taxonomy:
                    taxonomy[name] = self.taxonomy[name]
                else:
                    print('Not all names contain associated taxonomy, will not return taxonomy information')
                    taxonomy = {}
                    hasTax = False
            if hasSamples:
                if name in self.samples:
                    samples[name] = self.samples[name]
                else:
                    print('Not all names contain associated samples, will not return sample information')
                    samples = {}
                    hasSamples = False
        res = SeqData(sequences = sequences, taxonomy = taxonomy, samples = samples, pair_delim = self.pair_delim)
        res.get_pairings()
        return res

