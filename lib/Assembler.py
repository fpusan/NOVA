from collections import defaultdict
from itertools import combinations
from math import floor, ceil
from multiprocessing import Pool
from hashlib import sha1
from math import comb, ceil
from sys import maxsize
from functools import reduce

from lib.Graph import DBG
from lib.Overlap import overlap, getIdx, check_bimera

#overlap = profile(overlap)
import graph_tool as gt
from graph_tool.all import Graph

import numpy as np
import graph_tool as gt

from scipy.optimize import nnls


from datetime import datetime
import resource


#- numpy 1.20 will let us indicate dtype in numpy.concatenate instead of ugly hacks
#- remove seqs2names?
#- remove sets from competitive filter / rewrite better filter
#- track all reads (not only self.DBG.seqPaths) when calculating unrecovered and printing final message
#- revisar límites generados por check_bimera, podrían estar mal
#- distinguir seqPaths (seq:path dict) y seqPaths (path the secuencias en vez de vértices) en la terminología
#- use path hashes for everything rather than string hashes (right bow bimeras still use seq hashes)
#- memory peak can increase when calculating sequence abundances. Can we free memory right before?

def print_time(msg, end = None):
    dt = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    mem = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    print(f'{dt}\t{mem}MB\t{msg}', end = end)


class Assembler:

    multiprocessing_globals = tuple()

    @classmethod
    def set_multiprocessing_globals(cls, *args):
        # Multiprocessing.Pool.map comes with a significant overhead in serializing the arguments and sending them to the workers, which beats the purpose of multiprocessing.
        # What we'll do instead is store the required variables as a class attribute before starting the pool. Those will be copied when forking the process, which is much faster.
        cls.multiprocessing_globals = args

    @classmethod
    def clear_multiprocessing_globals(cls):
        cls.multiprocessing_globals = tuple()

    @classmethod
    def get_multiprocessing_globals(cls):
        return cls.multiprocessing_globals if len(cls.multiprocessing_globals) > 1 else cls.multiprocessing_globals[0]

    
    def __init__(self, seqData, pe_support_threshold, processors, sample, taxon, output_dir):
        self.names2seqs = seqData.sequences
        self.pairings = seqData.pairings
        self.seqPairings = {}
        for p1, p2 in self.pairings.items():
            s1 = self.names2seqs[p1]
            if s1 not in self.seqPairings:
                self.seqPairings[s1] = set()
            if p2:
                self.seqPairings[s1].add(self.names2seqs[p2])
                
        self.pe_support_threshold = pe_support_threshold
        self.processors = processors
        self.sample = sample
        self.taxon = taxon
        self.output_dir = output_dir
        
    def run(self, ksize):

        if len(self.names2seqs) < 100:
            print_time(f'{self.sample}; {self.taxon}\tLess than 100 sequences, ignoring')
            return {}, 0


        ### Map sequences to sequence names
        self.seqs2names = defaultdict(set)
        for name, seq in self.names2seqs.items():
           self.seqs2names[seq].add(name)


        ### Create DBG
        print_time(f'{self.sample}; {self.taxon}\tCreating DBG of k={ksize} from {len(self.names2seqs)} sequences')
        self.DBG = DBG(list(self.names2seqs.values()), ksize)
        msg = f'{self.sample}; {self.taxon}\t{self.DBG.includedSeqs} out of {len(self.names2seqs)} reads ({round(100*self.DBG.includedSeqs/len(self.names2seqs), 2)}%) were included in the graph'
        if self.DBG.nSS:
            msg += f', {self.DBG.nSS} sequences were smaller than {ksize} bases'
        print_time(msg)
        if self.DBG.nBS:
            print_time(f'{self.sample}; {self.taxon}\t{self.DBG.nBS} sequences ({self.DBG.nBV} vertices and {self.DBG.nBE} edges) contained low coverage kmers')
        print_time(f'{self.sample}; {self.taxon}\t{self.DBG.nvertices} vertices, {self.DBG.nedges} edges, {self.DBG.nsources} sources, {self.DBG.nsinks} sinks, {self.DBG.nsplitters} splitters, {self.DBG.nmergers} joiners, {self.DBG.npivots} pivots, {self.DBG.nsignature} signature ({round(100*self.DBG.nsignature/self.DBG.nvertices,2)}%)')

        ### Get signature sequences
        print_time(f'{self.sample}; {self.taxon}\tSummarizing {self.DBG.includedSeqs} sequences')
        self.signSeqPaths, correspondences = self.get_signature_sequences(self.DBG.seqPaths, merge_nosign = True)
        print_time(f'{self.sample}; {self.taxon}\t{self.DBG.includedSeqs} sequences were summarized in {len(self.signSeqPaths)} signature sequences')
        self.signSeqPairings = defaultdict(set)
        for n1, n2 in self.pairings.items():
            if n2:
                s1, s2 = self.names2seqs[n1], self.names2seqs[n2]
                if s1 in self.DBG.seqPaths and s2 in self.DBG.seqPaths: # They may be too short
                    c1, c2 = correspondences[s1], correspondences[s2]
                    assert c1 and c2
                    self.signSeqPairings[c1].add(c2)
        if self.output_dir:
            self.DBG.write_seqs([path for path in self.signSeqPaths.values()], f'{self.output_dir}/{self.taxon}/{self.sample}.{self.taxon}.signs.fasta')
            with open(f'{self.output_dir}/{self.taxon}/{self.sample}.{self.taxon}.signs.pairings.tsv', 'w') as outfile:
                for k1, pairs in self.signSeqPairings.items():
                    h1 = self.DBG.get_hash(self.signSeqPaths[k1])
                    pairs = '' if not pairs else ','.join([self.DBG.get_hash(self.signSeqPaths[k2]) for k2 in pairs])
                    outfile.write(f'{h1}\t{pairs}\n')
                    

        ### Get seed seqs joining paired end reads
        pairs2assemble = {}
        addedSignsPaired = set()
        addedSeqPairs = {}
        i = 0
        overlapping = []
        
        for x1, pairs in self.signSeqPairings.items():
            for x2 in pairs:
                if self.signSeqPaths[x1][0] > self.signSeqPaths[x2][0]:
                    c1, c2 = x2, x1
                else:
                    c1, c2 = x1, x2
                if (c1, c2) not in addedSignsPaired:
                    addedSignsPaired.add( (c1, c2) )
                    addedSeqPairs[i] = (c1, c2)
                    p1, p2 = self.signSeqPaths[c1], self.signSeqPaths[c2]
                    ol = overlap(p1, p2)
                    if ol.shape[0]: # This might happen since we have extended while getting signature sequences
                        overlapping.append(ol)
                        continue
                    pairs2assemble[i] = (p1, p2)
                    i+=1
                
        if not pairs2assemble and not overlapping:
            print_time(f'{self.sample}; {self.taxon}\tFound no valid pairs for assembly, ignoring') # We could still pull through if there are no (or few) signature kmers (just get paths from source to sink)
            return {}, 0

        singletons = {self.DBG.get_hash(path): path for seq, path in self.signSeqPaths.items() if seq not in {seq for pair in addedSeqPairs.values() for seq in pair}}
        print_time(f'{self.sample}; {self.taxon}\tFinding seed sequences from {len(pairs2assemble) + len(overlapping)} signature pairs and {len(singletons)} signature singletons')
        MAX_DEPTH = 400 ### THIS IS IMPORTANT. IF IT'S TOO LOW WE'LL LOSE VALID PAIRS
        self.set_multiprocessing_globals(  self.DBG, pairs2assemble, MAX_DEPTH )
        if self.processors == 1:
            joins = tuple(map(self.get_paths, pairs2assemble.keys()))
        else:
            with Pool(self.processors) as pool:
                joins = list(pool.imap(self.get_paths, pairs2assemble.keys(), chunksize=100))
        self.clear_multiprocessing_globals()

        addedSeqs = set()
        seedPaths = []
        for i, paths in enumerate(joins):
            if paths:
                addedSeqs.update(addedSeqPairs[i])
                seedPaths.extend(paths)
        
        seedSeqPaths = {self.DBG.get_hash(path): path for path in seedPaths}
        seedSeqPaths.update({self.DBG.get_hash(path): path for path in overlapping})
        unjoined = {seq: path for seq, path in self.signSeqPaths.items() if seq not in addedSeqs} # unjoined includes singletons AND paired end sequences that were not joined
        #seedSeqPaths.update(unjoined)
        seedSeqPaths.update(singletons)

        print_time(f'{self.sample}; {self.taxon}\tSummarizing {len(seedSeqPaths)} seed sequences')
        seedSeqPaths, _ = self.get_signature_sequences(seedSeqPaths, merge_nosign = False)
        addedPaths = set()
        
        if not seedSeqPaths:
            print_time(f'{self.sample}; {self.taxon}\tFound no seed sequences, ignoring')
            return {}, 0

        if self.output_dir:
            self.DBG.write_seqs([path for path in seedSeqPaths.values()], f'{self.output_dir}/{self.taxon}/{self.sample}.{self.taxon}.seeds.fasta')

##        prefix = '/home/fpuente/zobel/Projects/smiTE/testsNOVA/Freshwaters/mock4/16S/testU10/All_taxa/S_0.All_taxa'
##        self.signSeqPaths = {}
##        self.signSeqPairings = defaultdict(set)
##        seedSeqPaths = {}
##        for seq in open(prefix+'.signs.fasta').read().strip().lstrip('>').split('>'):
##            name, seq = seq.split('\n',1)
##            name = name.split('\t')[0]
##            seq = seq.replace('\n','').replace('N','').replace('-','').replace('.','')
##            path = self.DBG.seq2varray(seq)
##            self.signSeqPaths[self.DBG.get_hash(path)] = path
##        for line in open(prefix+'.signs.pairings.tsv'):
##            k1, pairs = line.strip().split('\t')
##            self.signSeqPairings[k1] = set(pairs.split(','))
##        for seq in open(prefix+'.seeds.fasta').read().strip().lstrip('>').split('>'):
##            name, seq = seq.split('\n',1)
##            name = name.split('\t')[0]
##            seq = seq.replace('\n','').replace('N','').replace('-','').replace('.','')
##            path = self.DBG.seq2varray(seq)
##            seedSeqPaths[self.DBG.get_hash(path)] = path

        finalJoinedSeqPaths, _ = self.iterative_join(seedSeqPaths, pe_cutoff = 0.9, val_with_sign = True, verbose = True)                    

        ### After we finish joining, check whether they are complete
        fullSeqPaths = {}
        incompletes = {}
        
        for key, path in finalJoinedSeqPaths.items():
            if path[0] in self.DBG.sources and path[-1] in self.DBG.sinks:
                fullSeqPaths[key] = path
            else:
                incompletes[key] = path
##        subsets = set()
##        for key, path in incompletes.items():
##            for path2 in fullSeqPaths.values():
##                if overlap(path, path2).shape[0]:
##                    subsets.add(key)
##                    break
##        incompletes = {key: path for key, path in incompletes.items() if key not in subsets}
                    
        #incompletes = {key: path for key, path in incompletes.items() if len(path) > 300-ksize+1}

        avgLen = round(np.mean([len(path)+ksize-1 for path in incompletes.values()]), 2) if incompletes else 0

        print_time(f'{self.sample}; {self.taxon}\tFound {len(fullSeqPaths)} complete and {len(incompletes)} incomplete sequences (avg. len {avgLen})')

        if self.output_dir:
            self.DBG.write_seqs([path for path in incompletes.values()], f'{self.output_dir}/{self.taxon}/{self.sample}.{self.taxon}.incompletes.fasta')


        ### Calculate PE scores for candidates
        print_time(f'{self.sample}; {self.taxon}\tFiltering {len(fullSeqPaths)} complete sequences')
        scores = self.get_pe_scores(fullSeqPaths, self.signSeqPaths, self.signSeqPairings)
        
        self.clear_multiprocessing_globals()
  
        if self.output_dir:
            self.DBG.write_seqs(fullSeqPaths.values(), f'{self.output_dir}/{self.taxon}/{self.sample}.{self.taxon}.candidates.fasta', scores, 'score') # trusting that fullPaths and scores are in the same order

        passingCandidates = {key: path for (key, path), score in zip(fullSeqPaths.items(), scores) if score >= 1} # A final filter

        if not passingCandidates:
            print_time(f'{self.sample}; {self.taxon}\tNo candidates with perfect score, chilling out...')
            passingCandidates = {key: path for (key, path), score in zip(fullSeqPaths.items(), scores) if score >= 0.7} # A final filter

        if not passingCandidates:
             print_time(f'{self.sample}; {self.taxon}\tNo valid candidates were found, ignoring')
             return {}, 0

        
        ### Search for bimeras and run competitive filter
        fullSeqKmers = dict( (self.DBG.path2info(path, return_sequence = True) for path in passingCandidates.values()) )

        print_time(f'{self.sample}; {self.taxon}\tLooking for bimeras over {len(fullSeqKmers)} candidate sequences')

        idx2seq = {i: seq for i, seq in enumerate(fullSeqKmers)}
        idx2path = {i: info['varray'] for i, (seq, info) in enumerate(fullSeqKmers.items())}
        mappings = {seq: {key for key, path2 in self.signSeqPaths.items() if overlap(path2, info['varray']).shape[0]} for seq, info in fullSeqKmers.items()}
        idx2covs = {i: sum(self.validate_path_pe(fullSeqKmers[seq]['varray'], self.signSeqPaths, self.signSeqPairings, return_vertices = True)) for i, seq in enumerate(fullSeqKmers)}

        self.set_multiprocessing_globals(idx2seq, idx2path, fullSeqKmers, self.signSeqPaths, self.signSeqPairings, mappings, idx2covs)
        
        if self.processors == 1 or len(fullSeqKmers) < self.processors:
            bimeras = map(self.get_bimeras, idx2seq.keys())
        else:
            with Pool(self.processors) as pool:
                bimeras = pool.map(self.get_bimeras, idx2seq.keys())

        bimdict = {}
        for i, bims in enumerate(bimeras):
            if not bims:
                continue
            seq = idx2seq[i]
            hB = self.DBG.get_hash_string(seq)
            bimdict[seq] = list()
            for j, k in bims:
                seq1, seq2 = idx2seq[j], idx2seq[k]
                h1, h2 = self.DBG.get_hash_string(seq1), self.DBG.get_hash_string(seq2)
                bimdict[seq].append( (hB, h1, h2) )
            
        if self.output_dir:
            with open(f'{self.output_dir}/{self.taxon}/{self.sample}.{self.taxon}.candidates.bimeras.tsv', 'w') as outfile:
                for bims in bimdict.values():
                    for hB, h1, h2 in bims:
                        outfile.write(f'{hB}\t{h1},{h2}\n')         
        fullSeqKmers = {seq: info for seq, info in fullSeqKmers.items() if seq not in bimdict}
        
        print_time(f'{self.sample}; {self.taxon}\tRunning competitive filter over {len(fullSeqKmers)} candidate sequences')
        idx2fullSeqs = {i: seq for i, seq in enumerate(fullSeqKmers)}
        self.set_multiprocessing_globals( idx2fullSeqs, fullSeqKmers, self.signSeqPaths, self.signSeqPairings )
        if self.processors == 1 or len(fullSeqKmers) < self.processors:
            competitiveFilter = map(self.validate_path_pe_competitive, idx2fullSeqs.keys())
        else:
            with Pool(self.processors) as pool:
                competitiveFilter = pool.map(self.validate_path_pe_competitive, idx2fullSeqs.keys())
        fullSeqKmers = dict([(seq, info) for (seq, info), isGood in zip(fullSeqKmers.items(), competitiveFilter) if isGood])
        self.clear_multiprocessing_globals()

        ### Build equation system
        print_time(f'{self.sample}; {self.taxon}\tEstimating the abundance of {len(fullSeqKmers)} candidate sequences')
        x = np.empty( (len(self.DBG.kmer2vertex), len(fullSeqKmers) ), dtype=np.uint8)
        y = np.empty(len(self.DBG.kmer2vertex), dtype=np.uint32)

        for i, (kmer, abund) in enumerate(self.DBG.kmerAbund.items()):
            vertex = self.DBG.kmer2vertex[kmer]
            y[i] = abund
            eq = np.empty(len(fullSeqKmers), dtype=np.uint8)
            for j, (fs, fs_kmers) in enumerate(fullSeqKmers.items()): # assuming iteration order in dicts is stable which should be above 3.7
                if vertex in fs_kmers['vset']:
                    eq[j] = 1
                else:
                    eq[j] = 0
            x[i] = eq

##        x = np.empty( (len(self.DBG.edgeAbund), len(fullSeqKmers) ), dtype=np.uint8)
##        y = np.empty(len(self.DBG.edgeAbund), dtype=np.uint32)
##        for i, (edge, abund) in enumerate(self.DBG.edgeAbund.items()):
##            edge = {self.DBG.kmer2vertex[kmer] for kmer in edge}
##            y[i] = abund
##            eq = np.empty(len(fullSeqKmers), dtype=np.uint8)
##            for j, (fs, fs_kmers) in enumerate(fullSeqKmers.items()): # assuming iteration order in dicts is stable which should be above 3.7
##                if edge.issubset(fs_kmers['vset']):
##                    eq[j] = 1
##                else:
##                    eq[j] = 0
##            x[i] = eq
            


        ### Go for the eyes, boo!
        abunds, residual = nnls(x,y)

        for abund, info in zip(abunds, fullSeqKmers.values()):
            info['abund'] = abund

        if self.output_dir:
            self.DBG.write_seqs([info['varray'] for info in fullSeqKmers.values()], f'{self.output_dir}/{self.taxon}/{self.sample}.{self.taxon}.final.fasta', [info['abund'] for info in fullSeqKmers.values()], 'abund')

        ### Transform "sequence abundances" into read counts
        fullSeqCounts = defaultdict(float)
        totalReads = 0
        unrecoveredID = f'{self.taxon}.Unrecovered'
        
        for seq, path in self.DBG.seqPaths.items():
            counts = self.DBG.seqAbund[seq]
            totalReads += counts
            hits = set()
            for fullSeq, fsinfo in fullSeqKmers.items():
                if not fsinfo['abund']:
                    continue
                if overlap(path, fsinfo.varray).shape[0] > 0:
                    hits.add(fullSeq)
            if not hits:
                fullSeqCounts[unrecoveredID] += counts
            else:
                hitsAbund = sum([fullSeqKmers[hit]['abund'] for hit in hits])
                for hit in hits:
                    fullSeqCounts[hit] += counts * fullSeqKmers[hit]['abund'] / hitsAbund


        ### Round nicely
        floorSum = sum([floor(abund) for abund in fullSeqCounts.values()])
        remainders = {seq: abund - floor(abund) for seq, abund in fullSeqCounts.items()}
        ceiled = 0 # We will "ceil" the sequences with the highest remainder and "floor" the rest so that their sum is exactly totalReads
        for seq, rem in sorted(remainders.items(), key = lambda x: x[1], reverse = True):
            if ceiled < totalReads - floorSum:
                fullSeqCounts[seq] = ceil(fullSeqCounts[seq])
                ceiled += 1
            else:
                fullSeqCounts[seq] = floor(fullSeqCounts[seq])

        nfulls = len([fullSeq for fullSeq in fullSeqCounts if fullSeq != unrecoveredID])
        countsInFulls = sum([counts for fullSeq, counts in fullSeqCounts.items() if fullSeq != unrecoveredID])
        percentAssigned = round(100*countsInFulls/totalReads, 2)
        print_time(f'{self.sample}; {self.taxon}\t{countsInFulls} out of {totalReads} reads ({percentAssigned}%) were assigned to {nfulls} variants')

        return fullSeqCounts, residual


    

    def get_signature_sequences(self, seqPaths, merge_nosign = True):
        sign2seqs = defaultdict(list) # {tuple_of_sorted_signature_vertices: seq}
        noSigns = {}
        # Group all the sequences sharing the same signature kmers
        for seq, path in seqPaths.items():
            sign = frozenset(path) & self.DBG.signature
            if sign:
                sign2seqs[sign].append(seq)
            else:
                noSigns[seq] = path

        # Combine sequences without no signature kmers. As long as they share one kmer, they can be merged
        if merge_nosign:
            joinedNoSigns, correspondences = self.iterative_join(noSigns, assume_nosign = True, verbose = False)
            for seq, cor in correspondences.items():
                try:
                    assert len(cor) == 1
                except:
                    print(seq)
                    for c in cor:
                        print(self.DBG.reconstruct_sequence(joinedNoSigns[c]))
                    raise
                correspondences[seq] = list(cor)[0]
        else:
            joinedNoSigns, correspondences = noSigns, {seq: seq for seq in noSigns}

        
        preKmers    = len({v for path in noSigns.values() for v in path})
        joinedKmers = len({v for path in joinedNoSigns.values() for v in path})
        assert preKmers == joinedKmers


        # Now reconstruct the longest possible sequence for each set of sequences
        signSeqPaths = {}
        for sign, seqs in sign2seqs.items():
            # All sequences here overlap (since they have the same signature kmers.
            # It is a matter to find the earliest starting one and the latest ending one and overlap them
            # This is trivial because of topological sort
            base = seqs[0]
            start = seqs[0]
            spos = seqPaths[start][0]
            end = seqs[0]
            epos = seqPaths[start][-1]
            for seq in seqs[1:]:
                spos2 = seqPaths[seq][0]
                epos2 = seqPaths[seq][-1]
                if spos2 < spos:
                    start = seq
                    spos = spos2
                if epos2 > epos:
                    end = seq
                    epos = epos2
            path = overlap(seqPaths[start], seqPaths[end])
            key = self.DBG.get_hash(path)
            signSeqPaths[key] = path
            for seq in seqs:
                correspondences[seq] = key
        
        signSeqPaths.update(joinedNoSigns)
        
        preKmers  = len({v for path in seqPaths.values() for v in path})
        signKmers = len({v for path in signSeqPaths.values() for v in path})
        assert preKmers == signKmers
            
        return signSeqPaths, correspondences


    def iterative_join(self, candidates, pe_cutoff = 0, val_with_sign = False, assume_nosign = False, verbose = True):

        res = {}
        idx2key  = {i: key  for i, key  in enumerate(candidates)}
        idx2path = {i: path for i, path in enumerate(candidates.values())}
        idx2sign = {i: frozenset(path) & self.DBG.signature for i, path in enumerate(candidates.values())}

        if True: #not pe_cutoff:
            # Perform initial join
            if verbose:
                print_time(f'{self.sample}; {self.taxon}\tJoining {len(candidates)} seed sequences')
            goodJoins = self.join_paths_dispatcher(idx2path, idx2sign, pe_cutoff = pe_cutoff, val_with_sign = val_with_sign, assume_nosign = assume_nosign)

            # Create sequence graph
            successors = defaultdict(set)
            predecessors = defaultdict(set)

            for i, j in goodJoins:
                p1, p2 = idx2path[i], idx2path[j]
                p1start, p1end = p1[0], p1[-1]
                p2start, p2end = p2[0], p2[-1]
                if p1start > p2start:
                    i, j = j, i
                    p1, p2 = p2, p1
                    p1start, p2start = p2start, p1start
                    p1end, p2end = p2end, p1end
                if p2end >= p1end and overlap(p1, p2).shape[0]:
                    successors[i].add(j)
                    predecessors[j].add(i)
                    
            G2 = Graph()
            G2.add_vertex(len(idx2path))
            for i, succs in successors.items():
                for j in succs:
                    G2.add_edge(i,j)
            assert gt.topology.is_DAG(G2)

            if True and self.output_dir:
                idx2pathHash = {i: self.DBG.get_hash(path) for i, path in idx2path.items()}
                with open(f'{self.output_dir}/{self.taxon}/{self.sample}.{self.taxon}.seeds.goodJoins.tsv', 'w') as outfile:
                    for i, succs in successors.items():
                        for j in succs:
                            outfile.write(f'{idx2pathHash[i]}\t{idx2pathHash[j]}\n')

        else:
            prefix = '/home/fpuente/zobel/Projects/smiTE/testsNOVA/Freshwaters/mock4/16S/testU10/All_taxa/S_0.All_taxa'
            goodJoins = []
            key2idx = {key: i for i, key in idx2key.items()}
            successors = defaultdict(set)
            predecessors = defaultdict(set)
            with open(prefix+'.seeds.goodJoins.tsv') as infile:
                for line in infile:
                    k1, k2 = line.strip().split('\t')
                    i, j = key2idx[k1], key2idx[k2]
                    successors[i].add(j)
                    predecessors[j].add(i)

        
        

        # Extend sequences
        origins = {i for i in idx2path if not predecessors[i]}
        key2sp = {} 
        self.set_multiprocessing_globals( idx2path, idx2sign, successors, assume_nosign )
        if verbose:
            print_time(f'{self.sample}; {self.taxon}\tExtending {len(origins)} origins')
        if self.processors == 1:
            extensions = map(self.extend_path, origins)
        else:
            pool = Pool(self.processors)
            extensions = pool.imap_unordered(self.extend_path, origins, chunksize = 1) #ceil(len(origins)/(self.processors*4)))
        for i, sps in enumerate(extensions):
            #if verbose:
            #    if not i%10:
            #        percent = round(100*i/len(origins),1)
            #        print_time(f'{self.sample}; {self.taxon}\tExtending {len(origins)} origins ({percent}%)', end = '\r')
            for sp in sps:
                path = self.path_from_ids(sp, idx2path)
                assert path.shape[0] ####### ASSERTION ###
                hash_ = DBG.get_hash(path)
                res[hash_] = path
                key2sp[hash_] = sp
        #if verbose:
        #    print_time(f'{self.sample}; {self.taxon}\tExtending {len(origins)} origins (100%)   ')
        
        if self.processors > 1:
            pool.close()


##        # Remove subpaths
##        subpaths = set()
##        for k1, k2 in combinations(res, 2):
##            p1, p2 = res[k1], res[k2]
##            if overlap(p1, p2).shape[0]:
##                if p1.shape[0] > p2.shape[0]:
##                    subpaths.add(k2)
##                else:
##                    subpaths.add(k1)
##        res = {key: path for key, path in res.items() if key not in subpaths}


##        for key1, path1 in res.items(): ####################### ASSERTIONS: check that no result path extend another result path, and that no input sequence extends a result path (WILL FAIL IF THERE ARE SUBPATHS)
##            for path2 in res.values():
##                ol = overlap(path1, path2)
##                if ol.shape[0]:
##                    if ol.shape[0] != max([path1.shape[0], path2.shape[0]]):
##                        assert False
##            for path2 in candidates.values():
##                ol = overlap(path2, path1)
##                if ol.shape[0]:
##                    if ol.shape[0] != max([path1.shape[0], path2.shape[0]]):
##                        assert False

        # Calculate correspondences and add non-joined sequences
##        correspondences = {key: {key2 for key2, path2 in res.items() if overlap(path, path2).shape[0]} for key, path in candidates.items()}
        correspondences = defaultdict(set)
        for key, sp in key2sp.items():
            sp = set(sp)
            for i in sp:
                for j in predecessors[i]:
                    for k in successors[j]:
                        if k in sp:
                            correspondences[idx2key[j]].add(key)
                            break
                for j in successors[i]:
                    for k in predecessors[j]:
                        if k in sp:
                            correspondences[idx2key[j]].add(key)
                            break
        for key, path in candidates.items():
            if not correspondences[key]:
                res[key] = path
                correspondences[key].add(key)
        return res, correspondences
    


    @classmethod
    def extend_path(cls, o):
        idx2path, idx2sign, successors, assume_nosign = cls.get_multiprocessing_globals()
        sps = []
        addedPaths = set()
        exts = set([(o,)])
        while True:
                
            succs = defaultdict(set)
            
            for ext in exts:
                l = ext[-1]
                ss = set()
                for s in successors[l]:
                    if idx2path[s][-1] > idx2path[l][-1]:
                        ss.add(s)
                succs[ext] = ss

            exts = set()
            for sp,  ss in succs.items():
                if not ss:
                    pt = tuple(cls.path_from_ids(sp, idx2path))
                    if pt not in addedPaths:
                        sps.append(sp)
                else:
                    for s in ss:
                        exts.add(sp + (s,))                                
            if not exts:
                break
            k2exts, _ = cls.summarize_joins(exts, idx2path, idx2sign, assume_nosign = assume_nosign) # assume_nosign can be removed from summarize_joins since now we precalculate signs?
            exts = k2exts.values()
            sps = list( cls.summarize_joins(sps, idx2path, idx2sign, assume_nosign = assume_nosign)[0].values())

        return sps
            


    @classmethod
    def summarize_joins(cls, joins, idx2path, idx2sign, assume_nosign = False, simplify_pairs = False):
        addedPaths = set()
        key2origin = defaultdict(set)
        key2joins = defaultdict(set)
        origin2key = defaultdict(set)
        noSign2key = {} 
        ns_key = 0
        for ids in joins: # ids contain ids to the (two or more) seed sequences to be joined
            ids = tuple(sorted(ids, key=lambda i: idx2path[i][0]))
            #path = cls.path_from_ids(ids, idx2path)
            #  if not path.shape[0]:
            #      continue
            if assume_nosign:
                sign = []
            else:
                sign = frozenset().union(*[idx2sign[i] for i in ids])
            if sign:
                key = sign
            else: # nosign paths will have an integer key, which will propagate to new paths as long as they share an origin with the ones already included in that key
                existing_keys = {noSign2key[i] for i in ids if i in noSign2key}
                if not existing_keys:
                    key = ns_key
                    ns_key += 1
                elif len(existing_keys) == 1:
                    key = next(iter(existing_keys))
                else:
                    key = next(iter(existing_keys))
                    ids = [i for kP in existing_keys for i in key2origin[kP]]                        
                    for kP in existing_keys:
                        del key2origin[kP]
                    #for i in ids:
                    #    del origin2key[i]
                for i in ids:
                    noSign2key[i] = key

            key2origin[key].update(ids)
            key2joins[key].add(ids)
            #for i in ids:
            #    origin2key[i].add(key)

        key2joinedPath = {}
        addedPaths = set()
        key2seqPath = {}
        id2goodIds = defaultdict(set)
        for key, ids in key2origin.items(): # try to join the largest starter with the largest finisher
            #ids = sorted(ids, key=lambda i: idx2path[i][0]) # sort them so they are joinable in order (ONLY NEEDED FOR THE ASSERTION!)
            v0 = min([idx2path[i][0 ] for i in ids])
            vx = max([idx2path[i][-1] for i in ids])
            i0 = sorted([i for i in ids if idx2path[i][0 ]==v0], key = lambda i: len(idx2path[i]), reverse = True)[0]
            ix = sorted([i for i in ids if idx2path[i][-1]==vx], key = lambda i: len(idx2path[i]), reverse = True)[0]
            px = idx2path[ix]
            px0 = px[0]
            goodIds = [i0]
            while True:
                il = goodIds[-1]
                if il == ix:
                    break
                pl = idx2path[il]
                pl0 = pl[0]
                plx = pl[-1]
                if plx >= px0:
                    goodIds.append(ix)
                    break
                bestExt = (-1, plx)
                for i in ids: # add the longest extension
                    pi = idx2path[i]
                    if pi[0] <= plx:
                        pix = pi[-1]
                        if pix > bestExt[1]:
                            bestExt = (i, pix)
                assert bestExt[0] >= 0
                goodIds.append(bestExt[0])
            
            #path = cls.path_from_ids(goodIds, idx2path)
            #assert path.shape[0]
            #assert path.shape[0] ==  cls.path_from_ids(ids, idx2path).shape[0]
            for i in ids:
                pi = idx2path[i]
                for g in goodIds:
                    if i == g or (i,g) in ids: #overlap(pi, idx2path[g]).shape[0]:
                        id2goodIds[i].add(g)

            #key2joinedPath[key] = path
            key2seqPath[key] = tuple(goodIds)

        if simplify_pairs:
            key2goodJoins = defaultdict(set)
            for key, joins in key2joins.items():
                for i, j in joins:
                    goodIs = id2goodIds[i]
                    goodJs = id2goodIds[j]
                    for gi in goodIs:
                        for gj in goodJs:
                            if (gi, gj) in joins:
                                key2goodJoins[key].add( (gi, gj) )
            key2joins = key2goodJoins

        return key2seqPath, key2joins
            
        
        

    @staticmethod
    def path_from_ids(ids, idx2path):
        return reduce(lambda p1,p2: overlap(p1,p2), (idx2path[i] for i in ids))
        

    def get_pe_scores(self, query_seqPaths, val_seqPaths, val_seqPairings, key2seqPath = {}):
        # if key2seqPath is provided, seqPaths will be translated to paths by the workers
        idx2path = {i: path for i, path in enumerate(query_seqPaths.values())}
        idx2seqPath = {i: sp for i, sp in enumerate(key2seqPath.values())}
        
        workDict = idx2seqPath if idx2seqPath else idx2path
        
        self.set_multiprocessing_globals( idx2path, val_seqPaths, val_seqPairings, idx2seqPath )
        if self.processors == 1 or len(workDict) < 50:
            scores = tuple(map(self.validate_path_pe_from_globals, workDict.keys()))
        else:
            with Pool(self.processors) as pool:
                scores = pool.map(self.validate_path_pe_from_globals, workDict.keys())
        return scores


##    def join_paths_dispatcher2(self, idx2path, idx2sign, pe_cutoff, val_with_sign = False, assume_nosign = False):
##        
##        val_seqPaths = self.signSeqPaths if val_with_sign else self.DBG.seqPaths
##        val_seqPairings = self.signSeqPairings if val_with_sign else self.seqPairings
##        
##        self.set_multiprocessing_globals( idx2path, val_seqPaths, val_seqPairings, 0, assume_nosign )
##        
##        if self.processors == 1 or len(idx2path) < 100:
##            goodJoins_pre = [ids for ids, res in zip(combinations(idx2path, 2), map(self.join_pair, combinations(idx2path, 2))) if res]
##        else:
##            with Pool(self.processors) as pool:
##                chunksize = min([ceil(comb(len(idx2path), 2) / (self.processors * 4)), 100000]) ### Too high results in high memory usage (and idle cores sometimes?), too low in large IPC overhead
##                goodJoins_pre = [ids for ids, res in zip(combinations(idx2path, 2), pool.imap(self.join_pair, combinations(idx2path, 2), chunksize=chunksize)) if res]
##        self.clear_multiprocessing_globals()
##
##        if pe_cutoff: ####### REDO THIS USING SEQPATHS
##            key2sp, key2joins = self.summarize_joins(goodJoins_pre, idx2path, idx2sign, assume_nosign = False)
##            scores = self.get_pe_scores(joinedSeqPaths, val_seqPaths, val_seqPairings)
##            goodJoins = [ids for key, score in zip(joinedSeqPaths, scores) for ids in key2joins[key] if score > pe_cutoff]
##        else:
##            goodJoins = goodJoins_pre
##            
##        return goodJoins
##
##
##    @classmethod
##    def join_pair(cls, ids, return_merged = False):
##        idx2path, val_seqPaths, val_seqPairings, pe_cutoff, assume_nosign = cls.get_multiprocessing_globals()
##        i, j = ids
##        joinedpath = overlap(idx2path[i], idx2path[j], assume_nosign = assume_nosign)
##        goodJoin = False
##        if joinedpath.shape[0]:
##            if not pe_cutoff or cls.validate_path_pe(joinedpath, val_seqPaths, val_seqPairings) >= pe_cutoff:
##               goodJoin = True
##            elif return_merged:
##                joinedpath = np.empty(0, dtype=np.uint32)
##        return joinedpath if return_merged else goodJoin


    def join_paths_dispatcher(self, idx2path, idx2sign, pe_cutoff, val_with_sign = False, assume_nosign = False):
        
        idx2chunk = {i: chunk for i, chunk in enumerate(np.array_split(tuple(idx2path), self.processors * 20))} ### how does this chunk size affect memory usage?
        val_seqPaths = self.signSeqPaths if val_with_sign else self.DBG.seqPaths
        val_seqPairings = self.signSeqPairings if val_with_sign else self.seqPairings
        
        self.set_multiprocessing_globals( idx2chunk, idx2path, idx2sign, val_seqPaths, val_seqPairings, pe_cutoff, assume_nosign )

    
        if self.processors == 1 or len(idx2path) < 100:
            goodJoinsPre = (ids for idss in map(self.join_pairs, idx2chunk) for ids in idss)
        else:
            pool = Pool(self.processors)
            goodJoinsPre = (ids for idss in pool.imap_unordered(self.join_pairs, idx2chunk) for ids in idss)
        self.clear_multiprocessing_globals()

        if pe_cutoff:
            key2sp, key2joins = self.summarize_joins(goodJoinsPre, idx2path, idx2sign, assume_nosign = assume_nosign, simplify_pairs = True)
            scores = self.get_pe_scores(idx2path, val_seqPaths, val_seqPairings, key2seqPath = key2sp) ### removing this filter will likely have a minor effect in the result and save a chunk of time
            goodJoins = [ids for key, score in zip(key2sp, scores) for ids in key2joins[key] if score > pe_cutoff] ### since each worker also performs a filter
            #goodJoins = [ids for joins in key2joins.values() for ids in joins]
        else:
            goodJoins = goodJoinsPre

        if self.processors == 1 or len(idx2path) < 100:
            pool.close()
        
        return goodJoins


    @classmethod
    def join_pairs(cls, chunk):
        idx2chunk, idx2path, idx2sign, val_seqPaths, val_seqPairings, pe_cutoff, assume_nosign = cls.get_multiprocessing_globals()
        goodJoinsPre = []
        #joinedPaths = []
        for i in idx2chunk[chunk]:
            cjoins = [j for j in idx2path if j > i]
            res = ( ( (i,j), overlap(idx2path[i], idx2path[j], assume_nosign = assume_nosign) ) for j in cjoins)
            res = [ (ids, path) for ids, path in res if path.shape[0]]
            goodJoinsPre.extend( (info[0] for info in res) )
            #joinedPaths.extend( (info[1] for info in res) )
        if pe_cutoff:
            key2sp, key2joins = cls.summarize_joins(goodJoinsPre, idx2path, idx2sign, assume_nosign = assume_nosign, simplify_pairs = True)
            scores = (cls.validate_path_pe(cls.path_from_ids(sp, idx2path), val_seqPaths, val_seqPairings) for sp in key2sp.values())
            goodJoins = [ids for key, score in zip(key2sp, scores) for ids in key2joins[key] if score > pe_cutoff]
            #goodJoins = [ids for joins in key2joins.values() for ids in joins]
        else:
            goodJoins = goodJoinsPre
        
        return goodJoins



    @classmethod
    def get_paths(cls, i):
        DBG, pairs2assemble, maxDepth = cls.get_multiprocessing_globals()
        source, sink = pairs2assemble[i]
        leading = source[:-1]
        trailing = sink[1:]
        paths = (path for path in gt.topology.all_paths(DBG.G, source[-1], sink[0], cutoff=maxDepth))
        paths = [np.array(np.concatenate([leading, path, trailing]), dtype=np.uint32) for path in paths]
        paths = [path for path in paths if cls.validate_path_se(path, DBG.seqPaths)]                     
        return paths

    
    @classmethod
    def validate_path_se(cls, path, seqPaths, allow_partial = False):
        pathKmerSet     = set(path)
        seenKmers       = set()

        for path2 in seqPaths.values():
            if overlap(path2, path).shape[0]:
                seenKmers.update(path2)

        if allow_partial:
            return len(seenKmers) > 0
        else:
            return pathKmerSet.issubset(seenKmers) # there are paths in seenKmers that are not from this path, but it's faster this way


    @classmethod
    def validate_path_pe(cls, path, seqPaths, seqPairings, force = False, return_vertices = False):
        pathKmerSet = set(path)

        seenKmersPaired = set()
        seenKmersPairedDict = {v: 0 for v in path}
        confirmedKmersPaired = set()
        confirmedKmersPairedDict = {v: 0 for v in path}

        mappedNames = set()

        for key, path2 in seqPaths.items():
            if overlap(path2, path).shape[0]:
                mappedNames.add(key)

        mappedPairs = 0
        totalPairs = 0
        for pair1 in mappedNames:
            pairs = seqPairings[pair1]
            if pairs:
                totalPairs += 1
                seenKmersPaired.update(seqPaths[pair1])
                if return_vertices:
                    for v in seqPaths[pair1]:
                        seenKmersPairedDict[v] += 1
                for pair2 in pairs:
                    if pair2 in mappedNames:
                        mappedPairs += 1
                        confirmedKmersPaired.update(seqPaths[pair1])
                        confirmedKmersPaired.update(seqPaths[pair2])
                        if return_vertices:
                            for v in seqPaths[pair1]:
                                if v in confirmedKmersPairedDict:
                                    confirmedKmersPairedDict[v] += 1
        if not totalPairs:
            return 0
        
        if return_vertices:
            res = {}
            for v in seenKmersPairedDict:
                if not seenKmersPairedDict[v]:
                    res[v] = 0
                else:
                    res[v] = confirmedKmersPairedDict[v] / seenKmersPairedDict[v]
            return res
        if force:
            score = len(confirmedKmersPaired & pathKmerSet) / len(pathKmerSet)
        else:
            score = len(confirmedKmersPaired & seenKmersPaired) / len(seenKmersPaired)
        return score


    @classmethod
    def validate_path_pe_from_globals(cls, i):
        idx2path, seqPaths, seqPairings, idx2seqPath = cls.get_multiprocessing_globals()
        if idx2seqPath: # treat inputs as seqPaths
            path = cls.path_from_ids(idx2seqPath[i], idx2path)
        else:
            path = idx2path[i]
        return cls.validate_path_pe(path, seqPaths, seqPairings)


    @classmethod
    def get_bimeras(cls, i):
        idx2seq, idx2path, fullSeqKmers, signSeqPaths, signSeqPairings, mappings, idx2covs = cls.get_multiprocessing_globals()
        B = idx2path[i]
        seq = idx2seq[i]
        info = fullSeqKmers[seq]
        bims = []
        CB = idx2covs[i]
        lCands = {j for j, path in idx2path.items() if path[0] == B[0] and j != i}
        rCands = {k for k, path in idx2path.items() if path[-1] == B[-1] and k != i}

        for j in lCands:
            for k in rCands:     
                p1, p2 = idx2path[j], idx2path[k]
                C1, C2 = idx2covs[j], idx2covs[k]
                if CB > C1 or CB > C2:
                    continue
                bim_limits = check_bimera(B, p1, p2)
                if len(bim_limits):
                    for k1 in mappings[seq]:
                        break_outer = False
                        if signSeqPaths[k1][0] <= bim_limits[0]:
                            for k2 in signSeqPairings[k1]:
                                if k2 in mappings[seq] and signSeqPaths[k2][-1] >=  bim_limits[1]: # maybe the intervale between bim_limits[0] and bim_limits[1] is too long
                                    break_outer = True
                                    break
                        if break_outer:
                            break
                    else:      
                        bims.append( (j,k) )
        return bims



    @classmethod
    def validate_path_pe_competitive(cls, i):
        idx2fullSeqs, fullSeqKmers, seqPaths, seqPairings = cls.get_multiprocessing_globals()
        seq = idx2fullSeqs[i]
        info = fullSeqKmers[seq]
        v1cover = cls.validate_path_pe(fullSeqKmers[seq]['varray'], seqPaths, seqPairings, return_vertices = True)
        isGood = True
        for seq2, info2 in fullSeqKmers.items():
            if seq == seq2:
                continue
            ol = info['vsignset'] & info2['vsignset']
            if ol and len(info['vsignset']) - len(ol) < 20:

                v2cover = cls.validate_path_pe(fullSeqKmers[seq2]['varray'], seqPaths, seqPairings, return_vertices = True)

                covs1 = [v1cover[v] for v in ol]
                covs2 = [v2cover[v] for v in ol]
                for c1, c2 in zip(covs1, covs2):
                    if c1 == 1 and c2 < 1:
                        break
                else:
                    for c1, c2 in zip(covs1, covs2):
                        if c1 < 1 and c2 == 1:
                            if sum(covs1) < sum(covs2) - 3:
                                isGood = False
                            break
                    if not isGood:
                        break

        return isGood
                





