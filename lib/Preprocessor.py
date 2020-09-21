from collections import defaultdict, OrderedDict
from copy import copy
from os import mkdir
from subprocess import run, DEVNULL

TAXRANKS = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

class Preprocessor:
    def __init__(self, fwd_reads, rev_reads, align_ref, align_tax, correction_level, processors, output_dir, logfile):

        self.logfile = logfile
        self.processors = processors
        self.pre_dir = '{}/preprocess'.format(output_dir)
        mkdir(self.pre_dir)
        
        ### Transform fastq to fasta.
        print('\tTransforming reads to the fasta format', file = self.logfile)
        fwd_reads_prefix = '.'.join(fwd_reads.split('/')[-1].split('.')[:-1])
        fwd_reads_fasta = self.pre_dir + '/' + fwd_reads_prefix + '.fasta'
        fwd_reads_qual = self.pre_dir + '/' + fwd_reads_prefix + '.qual'

        rev_reads_prefix = '.'.join(rev_reads.split('/')[-1].split('.')[:-1])
        rev_reads_fasta = self.pre_dir + '/' + rev_reads_prefix + '.fasta'
        rev_reads_qual = self.pre_dir + '/' + rev_reads_prefix + '.qual'
        run(['mothur "#fastq.info(fastq={}, outputdir={})"'.format(fwd_reads, self.pre_dir)], shell = True, stdout=DEVNULL, stderr=DEVNULL)
        run(['rm', fwd_reads_qual])
        run(['mothur "#fastq.info(fastq={}, outputdir={})"'.format(rev_reads, self.pre_dir)], shell = True, stdout=DEVNULL, stderr=DEVNULL)
        run(['rm', rev_reads_qual])


        ### Align fastas against silva.
        print('\tAligning reads against {}'.format(align_ref), file = self.logfile)
        run(['mothur "#align.seqs(fasta={}, reference={}, processors={}. outputdir={})"'.format(fwd_reads_fasta, align_ref, self.processors, self.pre_dir)], shell=True)
        run(['mothur "#align.seqs(fasta={}, reference={}, processors={}, outputdir={})"'.format(rev_reads_fasta, align_ref, self.processors, self.pre_dir)], shell=True)


        ### Read fasta and get paired info.
        print('\tCorrecting taxonomy', file = self.logfile)
        fwd_reads = self.fasta2dict(fwd_reads_fasta)
        rev_reads = self.fasta2dict(rev_reads_fasta)
        assert len(fwd_reads) == len(rev_reads)
        pairs = tuple(zip(fwd_reads, rev_reads)) # We can do this since they are ordered dicts.

        assert False
        ##########################
        

        
        
        # get paired info from read files
        

        # align them to silva, generate combined report

        readSeqs = fasta2dict('../samples/Oral3.mock1.allsamples.fasta')
        templateSeqs = fasta2dict('/home/natalia/Projects/natalia/DB/silva.nr_v132/silva.nr_v132.opti_mcc.0.05.align')


        templateTax = {}
        for line in open('/home/fer/DB/silva.nr.v132/silva.nr_v132.tax'):
            ref, tax = line.strip().split('\t')
            tax = tax.strip(';').split(';')
            templateTax[ref] = dict(zip(TAXRANKS, tax))
        templateTax['no_match'] = {rank: 'Unclassified' for rank in TAXRANKS}

        tax2templates = defaultdict(set)
        for template, tax in templateTax.items():
            tax2templates[tax[CORRECTION_LEVEL]].add(template)

        readTax = parse_report('Oral3.mock1.allsamples.unique.align.report')
        expand_names(readTax, 'Oral3.mock1.allsamples.16S.names')
        pairings, pairTax, artifacts = correct_tax_paired(readTax, CORRECTION_LEVEL)

        tax2reads = defaultdict(set)
        for read, info in readTax.items():
            tax2reads[info['tax'][CORRECTION_LEVEL]].add(read)

        ### AND CORRECT TAXONOMY!!

        ### Return alignment object!


    def load_files(self): ###
        pass


    def realign(self, correct_dir): ###
        """
        Re-align reads against templates containing only sequences from their corrected taxonomy.
        """
        for tax in tax2reads:
            reads = {header: seq for header, seq in readSeqs.items() if header in tax2reads[tax]}
            templates = {header: seq for header, seq in templateSeqs.items() if header in tax2templates[tax]}
            print(tax, len(reads), len(templates))
            dict2fasta(reads, 'reads.{}.fasta'.format(tax))
            dict2fasta(templates, 'templates.{}.fasta'.format(tax))
            run(['mothur "#align.seqs(fasta=reads.{}.fasta, reference=templates.{}.fasta, processors={})"'.format(tax,tax,self.processors)], shell=True)

        readFiles = ['reads.{}.fasta'.format(tax) for tax in tax2reads]
        templateFiles = ['template.{}.fasta'.format(tax) for tax in tax2reads]
        alignFiles = ['reads.{}.align'.format(tax) for tax in tax2reads]
        reportFiles = ['reads.{}.align.report'.format(tax) for tax in tax2reads]

        self.merge_reports(reportFiles, 'corrected.align.report')

        run(['cat'] + alignFiles, stdout = open('corrected.align','w'))
        run(['mothur "#filter.seqs(fasta=corrected.align, processors={})"'.format(self.processors)], shell = True)
        run(['rm', 'corrected.filter', 'corrected.align'])



        run(['rm', 'corrected.align.report.pre'])

        with open('corrected.names', 'w') as outfile: ### NOT NEEDED ANYMORE
            for read in readSeqs:
                outfile.write('{}\t{}\n'.format(read, read))


        for tax in tax2reads:
            run(['rm reads.{}.*'.format(tax)], shell=True)
            run(['rm templates.{}.*'.format(tax)], shell=True)

        run(['rm *logfile'], shell=True)


    
    @staticmethod
    def fasta2dict(fasta):
        res = OrderedDict()
        for seq in open(fasta).read().strip().lstrip('>').split('>'):
            header, seq = seq.split('\n',1)
            header = header.split('\t')[0]
            res[header] = seq.replace('\n','')#.replace('N','')
        return res


    @staticmethod
    def dict2fasta(dict_, outname):
        with open(outname, 'w') as outfile:
            for header, seq in dict_.items():
                outfile.write('>{}\n{}\n'.format(header, seq))


    @staticmethod
    def parse_report(align_report):
        res = {}
        with open(align_report) as infile:
           infile.readline()
           for line in infile:
               line = line.strip().split('\t')
               header, ref, qstart, qend, rstart, rend, riden = line[0], line[2], line[7], line[8], line[9], line[10], line[-1]
               rstart, rend, qstart, qend, riden = int(rstart), int(rend), int(qstart), int(qend), float(riden)
               alen = qend - qstart
               score = round(alen*riden/100)
               res[header] = {'rstart': rstart, 'rend': rend, 'alen': alen, 'riden': riden, 'score': score, 'tax': templateTax[ref]}
        return res


    @staticmethod
    def expand_names(taxdict, names_file):
        for line in open(names_file):
            rep, names = line.strip().split('\t')
            names = names.strip(',').split(',')
            for name in names:
                taxdict[name] = copy(taxdict[rep])

    @staticmethod
    def merge_reports(report_files, merged_file):
        with open(merged_file, 'w') as outfile:
            wrote_header = False
            for f in report_files:
                with open(f) as infile:
                    for line in infile:
                        if line.startswith('QueryName'):
                            if not wrote_header:
                                wrote_header = True
                            else:
                                continue
                        outfile.write(line)
                

##    def correct_tax_paired(taxdict, tax_correction_level = 'family'): ### SELF!
##        
##            pairs = defaultdict(list) ### This will be now calculated elsewhere. Also correct pair calculation in Alignment object!
##            for header in taxdict:
##                pair = header.split('_pair_')[0]
##                pairs[pair].append(header)
##
##
##            pairTax = defaultdict(lambda: defaultdict(int))
##            for reads in pairs.values():
##                if len(reads) == 1:
##                    continue
##                elif len(reads) == 2:
##                    pair1, pair2 = reads
##                    tax1 = taxdict[pair1]['tax'][tax_correction_level]
##                    tax2 = taxdict[pair2]['tax'][tax_correction_level]
##                    pairTax[tax1][tax2] += 1
##                    pairTax[tax2][tax1] += 1
##                else:
##                    raise Exception('wtf')
##
##            # We now consider that any taxon for which there was no pair such that both members were classified to that taxon is an artifact.
##            artifacts = {taxon for taxon in pairTax if taxon not in pairTax[taxon] or 'nclassified' in taxon or pairTax[taxon][taxon] / sum(pairTax[taxon].values()) < 0.1}
##
##            def blind_correct(tax):
##                c = 0
##                while True: # potentially infinite loop
##                    c+=1
##                    if c==10:
##                        newTax = tax
##                        break
##                    bestOptions = list(sorted(pairTax[tax], key = lambda x: pairTax[tax1][x], reverse = True))
##                    if not bestOptions:
##                        newTax = tax
##                        break
##                    bestOptionsNoArtifact = [t for t in bestOptions if t not in artifacts]
##                    if bestOptionsNoArtifact:
##                        newTax = bestOptionsNoArtifact[0]
##                        break
##                    else:
##                        tax = bestOptions[0] # search for a noartifact taxonomy starting from the best option we had in this tax.
##                        
##                res = {rank: 'CORRECTED' for rank in TAXRANKS}
##                res[tax_correction_level] = newTax
##                return res
##
##        for reads in pairs.values():
##            if len(reads) == 1:
##                single = reads[0]
##                tax = taxdict[single]['tax'][tax_correction_level]
##                if tax in artifacts:
##                    taxdict[single]['tax'] = blind_correct(tax)
##            else: # len(reads) == 2:
##                pair1, pair2 = reads
##                tax1 = taxdict[pair1]['tax'][tax_correction_level]
##                tax2 = taxdict[pair2]['tax'][tax_correction_level]
##                
##                if tax1 != tax2:
##                    if tax1 not in artifacts and tax2 in artifacts:
##                        goodTax =  taxdict[pair1]['tax']
##                    elif tax1 in artifacts and tax2 not in artifacts:
##                        goodTax =  taxdict[pair2]['tax']
##                    else: # Both are artifacts or both are non artifacts
##                        if taxdict[pair1]['score'] > taxdict[pair2]['score']:
##                            if tax1 in artifacts: # both artifacts
##                                goodTax = blind_correct(tax1)
##                            else:
##                                goodTax =  taxdict[pair1]['tax']
##                        elif taxdict[pair1]['score'] < taxdict[pair2]['score']:
##                            if tax1 in artifacts: # both artifacts
##                                goodTax = blind_correct(tax2)
##                            else:
##                                goodTax =  taxdict[pair2]['tax']
##                        else: # same score!
##                            if sum(pairTax[tax1].values()) > sum(pairTax[tax2].values()):
##                                if tax1 in artifacts: # both artifacts
##                                    goodTax = blind_correct(tax1)
##                                else:
##                                    goodTax =  taxdict[pair1]['tax']
##                            elif sum(pairTax[tax1].values()) < sum(pairTax[tax2].values()):
##                                if tax1 in artifacts: # both artifacts
##                                    goodTax = blind_correct(tax2)
##                                else:
##                                    goodTax =  taxdict[pair2]['tax']
##                            else: # same score and same abundance!
##                                print('This message being printed means that we can\'t obtain meaningful knowledge nor make truly informed decisions. Embracing nihilism...')
##                    taxdict[pair1]['tax'] =  goodTax
##                    taxdict[pair2]['tax'] =  goodTax
##                          
##        for reads in pairs.values():
##            if len(reads) == 2:
##                pair1, pair2 = reads
##                tax1 = taxdict[pair1]['tax'][tax_correction_level]
##                tax2 = taxdict[pair2]['tax'][tax_correction_level]
##                assert tax1==tax2
##
##        return pairTax
##



Preprocessor('/home/fer/Projects/smiTE/mock1/samples/test/Oral3.mock1.S_0.perfect.InSilicoSeq_R1.fastq',
             '/home/fer/Projects/smiTE/mock1/samples/test/Oral3.mock1.S_0.perfect.InSilicoSeq_R2.fastq',
             '/home/fer/DB/silva.nr.v132/silva.nr_v132.align',
             '/home/fer/DB/silva.nr.v132/silva.nr_v132.tax',
             'family',
             6,
             '/home/fer/Projects/smiTE/mock1/samples/test/output',
             None)
    


