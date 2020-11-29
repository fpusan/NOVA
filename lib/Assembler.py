from collections import defaultdict
from itertools import combinations
from math import floor, ceil
from multiprocessing import Pool
from hashlib import sha1
from math import comb, ceil
from sys import maxsize
from functools import reduce

from lib.Graph import DBG
from lib.Overlap import overlap, getIdx

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
#- remove sets from abundance calculation


def print_time(msg):
    dt = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    mem = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    print(f'{dt}\t{mem}MB\t{msg}')


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
                    h1 = self.DBG.get_hash_string(self.DBG.reconstruct_sequence(self.signSeqPaths[k1]))
                    pairs = '' if not pairs else ','.join([self.DBG.get_hash_string(self.DBG.reconstruct_sequence(self.signSeqPaths[k2])) for k2 in pairs])
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

        singletons = {seq: path for seq, path in self.signSeqPaths.items() if seq not in {seq for pair in addedSeqPairs.values() for seq in pair}}
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
##        seedSeqPaths.update(unjoined)
        seedSeqPaths.update(singletons)
        print_time(f'{self.sample}; {self.taxon}\tSummarizing {len(seedSeqPaths)} seed sequences')
        seedSeqPaths, _ = self.get_signature_sequences(seedSeqPaths, merge_nosign = False)
        
        if not seedSeqPaths:
            print_time(f'{self.sample}; {self.taxon}\tFound no seed sequences, ignoring')
            return {}, 0

        if self.output_dir:
            self.DBG.write_seqs([path for path in seedSeqPaths.values()], f'{self.output_dir}/{self.taxon}/{self.sample}.{self.taxon}.seeds.fasta')



        finalJoinedSeqPaths, _ = self.iterative_join(seedSeqPaths, pe_cutoff = 0.9, val_with_sign = True, verbose = True)                    

        ### After we finish joining, check whether they are complete
        fullSeqPaths = {}
        incompletes = {}
        
        for key, path in finalJoinedSeqPaths.items():
            if path[0] in self.DBG.sources and path[-1] in self.DBG.sinks:
                fullSeqPaths[key] = path
            else:
                incompletes[key] = path
        subsets = set()
        for key, path in incompletes.items():
            for path2 in fullSeqPaths.values():
                if overlap(path, path2, subsets_as_overlaps=True).shape[0]:
                    subsets.add(key)
                    break
        incompletes = {key: path for key, path in incompletes.items() if key not in subsets}
                    
        #incompletes = {key: path for key, path in incompletes.items() if len(path) > 300-ksize+1}

        avgLen = round(np.mean([len(path)-ksize+1 for path in incompletes.values()]), 2) if incompletes else 0

        print_time(f'{self.sample}; {self.taxon}\tFound {len(fullSeqPaths)} complete and {len(incompletes)} incomplete sequences (avg. len {avgLen})')

        if self.output_dir:
            self.DBG.write_seqs([path for path in incompletes.values()], f'{self.output_dir}/{self.taxon}/{self.sample}.{self.taxon}.incompletes.fasta')


##        leftVertices = set()
##        rightVertices = set()
##        for path in incompletes.values():
##            leftVertices.add(path[0])
##            rightVertices.add(path[-1])
##        leftVertices = tuple(leftVertices)
##        rightVertices = tuple(rightVertices)
##            
##        # Extend kmers appearing at the start or end of incomplete sequences to sources and sinks
##        print_time(f'{self.sample}; {self.taxon}\tCalculating {len(leftVertices)} left and {len(rightVertices)} right extensions')
##
##        self.set_multiprocessing_globals( fullSeqPaths, incompletes, self.DBG, 1600, 'left' ) # Larger depth cutoff here bc we might want to extend long partial sequences that were not assembled
##        if self.processors == 1 or len(leftVertices) < self.processors:
##            leftExtensions = dict(zip(leftVertices, map(self.extend_vertex, leftVertices)))
##        else:
##            with Pool(self.processors) as pool:
##                leftExtensions = dict(zip(leftVertices, pool.map(self.extend_vertex, leftVertices)))
##        self.clear_multiprocessing_globals()
##
##        self.set_multiprocessing_globals( fullSeqPaths, incompletes, self.DBG, 1600, 'right' ) # Larger depth cutoff here bc we might want to extend long partial sequences that were not assembled
##        if self.processors == 1 or len(rightVertices) < self.processors:
##            rightExtensions = dict(zip(rightVertices, map(self.extend_vertex, rightVertices)))
##        else:
##            with Pool(self.processors) as pool:
##                rightExtensions = dict(zip(rightVertices, pool.map(self.extend_vertex, rightVertices)))
##        self.clear_multiprocessing_globals()
##
##        # Extend incomplete paths
##        print_time(f'{self.sample}; {self.taxon}\tExtending incomplete paths')
##        i = 0
##        idx2candidates = {}
##        idx2extensionVertices = {}
##        
##        for key, path in incompletes.items():
##
##            leftVertex = path[0]
##            rightVertex = path[-1]
##            leftPos = 0
##            rightPos = len(path) - 1 
##                
##            partials     = []
##            fullPaths    = []
##            
##            if rightVertex not in self.DBG.sinks: # Do right before so as to not mess up indexing
##                for ext in rightExtensions[rightVertex]:
##                    partials.append(np.concatenate( (path[:rightPos+1], ext) ) )
##            else:
##                partials.append(path)
##            
##            if leftVertex not in self.DBG.sources:
##                for partial in partials:
##                    for ext in leftExtensions[leftVertex]:
##                        fullPaths.append(np.concatenate( (ext, partial[leftPos:]) ) )
##            else:
##                fullPaths = partials
##
##            #assert fullPaths # Actually it might be that we found no valid extension for a particular left/right sign vertex...
##
##            fullPaths = [np.array(path, dtype=np.uint32) for path in fullPaths] # this won't be necessary after numpy 1.
##            for path in fullPaths:
##                assert path[0] in self.DBG.sources and path[-1] in self.DBG.sinks
##                idx2candidates[i] = path
##                idx2extensionVertices[i] = (leftVertex, rightVertex)
##                i+=1
##
##
##        # Filter out extended candidates
##        self.set_multiprocessing_globals( idx2candidates, idx2extensionVertices, self.DBG.vertex2paths )
##        if self.processors == 1 or len(idx2candidates) < 50:
##            fullPaths = [path for path, isValid in zip(idx2candidates.values(), map(self.validate_extension, idx2candidates.keys())) if isValid] ## self.validate_extension is failing now
##        else:
##            with Pool(self.processors) as pool:
##                fullPaths = [path for path, isValid in zip(idx2candidates.values(), pool.map(self.validate_extension, idx2candidates.keys())) if isValid]
##        self.clear_multiprocessing_globals()
##
##        fullSeqPaths.update( {self.DBG.get_hash(path): path for path in fullPaths} )
                


##        prefix = '/home/fpuente/zobel/Projects/smiTE/testsNOVA/Freshwaters/mock4/16S/testU/All_taxa/S_0.All_taxa'
##        self.signSeqPaths = {}
##        self.signSeqPairings = defaultdict(set)
##        fullSeqPaths = {}
##        for seq in open(prefix+'.signs.fasta').read().strip().lstrip('>').split('>'):
##            name, seq = seq.split('\n',1)
##            name = name.split('\t')[0]
##            seq = seq.replace('\n','').replace('N','').replace('-','').replace('.','')
##            key = self.DBG.get_hash_string(seq)
##            self.signSeqPaths[key] = self.DBG.seq2varray(seq)
##        for line in open(prefix+'.signs.pairings.tsv'):
##            k1, pairs = line.strip().split('\t')
##            self.signSeqPairings[k1] = set(pairs.split(','))
##        for seq in open(prefix+'.candidates.fasta').read().strip().lstrip('>').split('>'):
##            name, seq = seq.split('\n',1)
##            name = name.split('\t')[0]
##            seq = seq.replace('\n','').replace('N','').replace('-','').replace('.','')
##            path = self.DBG.seq2varray(seq)
##            key = self.DBG.get_hash(path)
##            fullSeqPaths[key] =  path

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

        

        fullSeqKmers = dict( (self.DBG.path2info(path, return_sequence = True) for path in passingCandidates.values()) )

        print_time(f'{self.sample}; {self.taxon}\tLooking for bimeras over {len(fullSeqKmers)} candidate sequences')
        from lib.bimeratest import is_bimera
        bimeras = defaultdict(list)
        covs = {seq: sum(self.validate_path_pe(fullSeqKmers[seq]['varray'], self.signSeqPaths, self.signSeqPairings, return_vertices = True)) for seq in fullSeqKmers}
        mappings = {seq: {key for key, path2 in self.signSeqPaths.items() if overlap(path2, info['varray']).shape[0]} for seq, info in fullSeqKmers.items()}

        for seq, info in fullSeqKmers.items():
            B = info['varray']
            CB = covs[seq]
            for seq1, seq2 in combinations(fullSeqKmers, 2):
                if seq1 == seq or seq2 == seq:
                    continue
                p1, p2 = fullSeqKmers[seq1]['varray'], fullSeqKmers[seq2]['varray']
                C1, C2 = covs[seq1], covs[seq2]
                if CB > C1 or CB > C2:
                    continue
                bim_limits = is_bimera(B, p1, p2)
                if len(bim_limits):
                    for k1 in mappings[seq]:
                        break_outer = False
                        if self.signSeqPaths[k1][0] <= bim_limits[0]:
                            for k2 in self.signSeqPairings[k1]:
                                if k2 in mappings[seq] and self.signSeqPaths[k2][-1] >=  bim_limits[1]: # maybe the intervale between bim_limits[0] and bim_limits[1] is too long
                                    break_outer = True
                                    break
                        if break_outer:
                            break
                    else:      
                        hB, h1, h2 = map(DBG.get_hash_string, (seq, seq1, seq2))
                        bimeras[seq].append( (hB, CB, h1, C1, h2, C2) )
                    
        if self.output_dir:
            with open(f'{self.output_dir}/{self.taxon}/{self.sample}.{self.taxon}.candidates.bimeras.tsv', 'w') as outfile:
                for bims in bimeras.values():
                    for hB, CB, h1, C1, h2, C2 in bims:
                        outfile.write(f'{hB}:{CB}\t{h1}:{C1},{h2}:{C2}\n')         
        fullSeqKmers = {seq: info for seq, info in fullSeqKmers.items() if seq not in bimeras}

        print_time(f'{self.sample}; {self.taxon}\tRunning competitive filter over {len(fullSeqKmers)} complete sequences')
        idx2fullSeqs = {i: seq for i, seq in enumerate(fullSeqKmers)}
        self.set_multiprocessing_globals( idx2fullSeqs, fullSeqKmers, self.signSeqPaths, self.signSeqPairings )
        if self.processors == 1 or len(fullSeqKmers) < self.processors:
            competitiveFilter = map(self.validate_path_pe_competitive, range(len(idx2fullSeqs)))
        else:
            with Pool(self.processors) as pool:
                competitiveFilter = pool.map(self.validate_path_pe_competitive, range(len(idx2fullSeqs)))
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
                if overlap(path, fsinfo.varray, subsets_as_overlaps = True).shape[0] > 0:
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


    def join2(self, candidates):
        res = {}
        idx2key = {i: key for i, key in enumerate(candidates)}
        paths, goodJoinsPre, singleIdx = self.join_paths_dispatcher(idx2key, candidates, pe_cutoff = 0)
        for i in singleIdx:
            key = idx2key[i]
            res[key] = candidates[key]
        goodJoins = goodJoinsPre
##        goodJoins = self.remove_redundant_edges(goodJoinsPre)
##        removed = {k for ids in goodJoinsPre for k in ids} - {k for ids in goodJoins for k in ids}
##        idx2key = {i: key for i, key in idx2key.items() if i not in removed}
##        candidates = {key: candidates[key] for key in idx2key.values()}
        res.update(self.graph_join(goodJoins, idx2key, candidates))
        return res



    def iterative_join(self, candidates, targets = [], pe_cutoff = 0, val_with_sign = False, assume_nosign = False, verbose = True):
        ### assume_nosign will use a faster algorithm for join calculation and consider subsets as overlaps


        toJoin = candidates
        res = {}
        idx2key = {i: key for i, key in enumerate(toJoin)}
        idx2key0 = {i: key for i, key in idx2key.items()}
        step = 0

        mappings = {0: {i: {i} for i in idx2key}}
        correspondences = defaultdict(set)

        ### Join seed paths
        while True:
            
            step += 1

            if len(toJoin) < 2:
                for i, key in idx2key.items():
                    path = toJoin[key]
                    hash_ = self.DBG.get_hash(path)
                    res[hash_] = path
                    for o in mappings[step-1][i]:
                        correspondences[idx2key0[o]].add(hash_)
                break


            if verbose:
                msg = f'{self.sample}; {self.taxon}\tSTEP {step}: Joining {len(toJoin)} seed sequences'
                if targets:
                    msg += f' ({len(targets)} joins)'
                else:
                    msg += ' (all vs all)'
                print_time(msg)


            joinedSeqKmers = {}
            signJoinedSeqKmers = {}

            pe_cutoff = pe_cutoff if step < 2 else 0

            # Join
            joinedPaths, goodJoinsPre, _ = self.join_paths_dispatcher(idx2key, toJoin, pe_cutoff = 0, targets = targets, val_with_sign = val_with_sign, subsets_as_overlaps = assume_nosign, assume_nosign = assume_nosign, direct_join = step > 1)

            # Summarize signature paths
            signs2paths, goodJoins, signSeq2origin, origin2signSeq = self.summarize_joins(joinedPaths, goodJoinsPre, self.DBG, None, None, pe_cutoff = 0, assume_nosign = assume_nosign)

            allJoined = {k for ids in goodJoins for k in ids}
            singleIdx = {i for i in idx2key if i not in allJoined}
            origin2signSeq = {k: signs for k, signs in origin2signSeq.items() if k in allJoined}

            # re-cast as a key:path dictionary
            for key, info in signs2paths.items():
                signs2paths[key] = info[0]

            # Filter sequences if required
            if pe_cutoff:
                print_time(f'{self.sample}; {self.taxon}\tSTEP {step}: Filtering {len(signs2paths)} candidates')
                scores = self.get_pe_scores(signs2paths, self.signSeqPaths, self.signSeqPairings)
                signs2paths = {key: info for (key, info), score in zip(signs2paths.items(), scores) if score >= pe_cutoff}
                keySet = set(signs2paths)
                goodJoins = [(i,j) for i, j in goodJoins if origin2signSeq[i] & origin2signSeq[j] & keySet]
                signSeq2origin = {key: ids for key, ids in signSeq2origin.items() if key in signs2paths}
                allJoined = {k for ids in goodJoins for k in ids}

                

            # If we couldn't join them in this step, assume we'll never can.
            for i, key in idx2key.items():
                if i in singleIdx:
                    path = toJoin[key]
                    hash_ = self.DBG.get_hash(path)
                    res[hash_] = path
                    for o in mappings[step-1][i]:
                        correspondences[idx2key0[o]].add(hash_)                      

            # If we joined them and they are not full, try to extend them in the next step
            toJoin = {}

            for key, path in signs2paths.items():
                if path[0] in self.DBG.sources and path[-1] in self.DBG.sinks:
                    hash_ = DBG.get_hash(path)
                    res[hash_] = path
                    for ori in signSeq2origin[key]:
                        for o in mappings[step-1][ori]:
                            correspondences[idx2key0[o]].add(hash_)
                else:
                    toJoin[key] = path

            idx2key = {i: key for i, key in enumerate(toJoin)}
            key2idx = {key: i for i, key in idx2key.items()}

            # Define join targets for the next iteration
            targets_pre = set()
            for keys in origin2signSeq.values():
                # keys are keys to paths that shared the same origin (thus they can be merged)
                for k1, k2 in combinations(keys, 2):
                    if k1 not in toJoin or k2 not in toJoin:
                        continue
                    p1, p2 = toJoin[k1], toJoin[k2]
                    if p1[0] > p2[0]:
                        k1, k2 = k2, k1
                        p1, p2 = p2, p1
                    if not assume_nosign and (p1[-1] >= p2[-1] or p1[0] == p2[0]): # if it begins before and ends after, ignore (we won't merge overlaps anyway, unless assuming nosign)
                        continue
                    i, j = key2idx[k1], key2idx[k2]
                    targets_pre.add( (i, j) )

            # Remove redundant edges
            if assume_nosign:
                targets = targets_pre
            else:
                targets = self.remove_redundant_edges(targets_pre) ## G2 is not a DAG in 200k test (when getting signature sequences)
                removed = {k for ids in targets_pre for k in ids} - {k for ids in targets for k in ids}
                idx2key = {i: key for i, key in idx2key.items() if i not in removed}
                toJoin = {key: toJoin[key] for key in idx2key.values()}
            if False: #not assume_nosign:
                paths = self.graph_join(targets, idx2key, toJoin)
                res.update(paths)
                return res, {}
            

            # Track mapping between sequences joined in the current iteration and the original sequences
            mappings[step] = {}
            for k in toJoin:
                origins = signSeq2origin[k]
                mappings[step][key2idx[k]] = {o for ori in origins for o in mappings[step-1][ori]}
   
        return res, correspondences



    @classmethod
    def remove_redundant_edges(cls, edges):
        sourcesDict = defaultdict(set)
        targetsDict = defaultdict(set)
        for i, j in edges:
            sourcesDict[j].add(i)
            targetsDict[i].add(j)

        # Identify articulation points in the sequence graph
        G2 = Graph()
        G2.add_vertex(len({v for ids in edges for v in ids}))
        for i, j in edges:
            G2.add_edge(i,j)
        assert gt.topology.is_DAG(G2)
        articulations = {v for v in gt.topology.label_biconnected_components(G2)[2]}
        
        # For each join target, check whether there is any path leading to both the origin and the destination
        goodEdges = []
        for i, j in edges:
            isGood = True
            if i not in articulations:
                for s in sourcesDict[j]:
                    if s in sourcesDict[i]: # If a second vertex leads to both the origin and the destination, then this edge is redundant and we remove it.
                        isGood = False
                        break

            if j not in articulations:
                for t in targetsDict[j]:
                    if i in sourcesDict[t]:
                        isGood = False
                        break
            if isGood:
                goodEdges.append( (i,j) )
                    
        return goodEdges



    def graph_join(self, edges, idx2key, toJoin):
        print_time('start GJ')
        G2 = Graph()
        G2.add_vertex(len({v for ids in edges for v in ids}))
        for i, j in edges:
            G2.add_edge(i,j)
        assert gt.topology.is_DAG(G2)
        sources = {int(v) for v in G2.vertices() if not list(v.in_neighbors()) } # We use integers rather than vertex objects bc vertex objects can't be pickled
        sinks = {int(v) for v in G2.vertices() if not list(v.out_neighbors())}
        paths = {}
        gt.openmp_set_num_threads(self.processors)
        for source in sources:
            dist_map, pred_map = gt.topology.shortest_distance(G2, source=source, pred_map=True, dag=True)
            for sink in sinks:
                for seqPath in gt.topology.all_shortest_paths(G2, source, sink, dist_map=dist_map, pred_map=pred_map):
                    path = reduce(lambda p1,p2: overlap(p1,p2, subsets_as_overlaps=True), (toJoin[idx2key[idx]] for idx in seqPath))
                    if path.shape[0]:
                        hash_ = self.DBG.get_hash(path)
                        paths[hash_] = path
                    
        print_time(f'end GJ ({len(paths)})')
        return paths
        

        



    @classmethod
    def summarize_joins(cls, joinedPaths, goodJoinsPre, DBG, val_seqPaths, val_pairings, pe_cutoff = 0, assume_nosign = False):

        signs2paths = {}
        addedPaths = set()
        goodJoins = []
        signSeq2origin = defaultdict(set)
        origin2signSeq = defaultdict(set)
        noSign2key = {} 
        ns_key = 0
        # here keys can be either sequences, hashes, signature vertices (for intermediate sign sequences) or arbitrary integers (for intermediate nosign sequences). They are treated indistinctly
        # we however make the point of adding only hashes as keys in the results dictionares (res, correspondences)
        for path, ids in zip(joinedPaths, goodJoinsPre):
            if not path.shape[0]:
                continue
            i, j = ids
            goodJoins.append(ids)
            hash_ = DBG.get_hash(path)
            if assume_nosign:
                sign = []
            else:
                sign = frozenset(path) & DBG.signature
            if sign:
                key = sign
            else: # nosign paths will have an integer key, which will propagate to new paths as long as they share an origin with the ones already included in that key
                if i in noSign2key:
                    key = noSign2key[i]
                    noSign2key[j] = key
                elif j in noSign2key:
                    key = noSign2key[j]
                    noSign2key[i] = key
                else:
                    key = ns_key
                    noSign2key[i] = key
                    noSign2key[j] = key
                    ns_key += 1
            signSeq2origin[key].update(ids)
            origin2signSeq[i].add(key)
            origin2signSeq[j].add(key)
            if hash_ in addedPaths:
                continue
            spos = path[0]
            epos = path[-1]
            if key not in signs2paths:
                signs2paths[key] = (path, spos, epos)
            else:
                pathP, sposP, eposP = signs2paths[key]
                if spos >= sposP and epos <= eposP: # contained in the previous best
                    pass
                elif spos < sposP and epos > eposP:
                    signs2paths[key] = (path, spos, epos) # contains the previous best
                elif spos < sposP: # left extension of the previous best
                    signs2paths[key] = (overlap(path, pathP), spos, eposP)
                elif epos > eposP: # right extension of the previous best
                    signs2paths[key] = (overlap(pathP, path), sposP, epos)
                else:
                    raise Exception('wtf')
            addedPaths.add(hash_)

        # Filter candidate joins
        if pe_cutoff and val_seqPaths and val_pairings:
            scores = [cls.validate_path_pe(info[0], val_seqPaths, val_pairings, force = False, return_vertices = False) for info in signs2paths.values()]
            signs2paths = {key: info for (key, info), score in zip(signs2paths.items(), scores) if score >= pe_cutoff}
            keySet = set(signs2paths)
            goodJoins = [(i,j) for i, j in goodJoins if origin2signSeq[i] & origin2signSeq[j] & keySet]
            signSeq2origin = {key: ids for key, ids in signSeq2origin.items() if key in signs2paths}
            allJoined = {k for ids in goodJoins for k in ids}


        return signs2paths, goodJoins, signSeq2origin, origin2signSeq



    def get_pe_scores(self, query_seqPaths, val_seqPaths, val_seqPairings):
        idx2paths = {i: path for i, path in enumerate(query_seqPaths.values())}
        self.set_multiprocessing_globals( idx2paths, val_seqPaths, val_seqPairings )
        if self.processors == 1 or len(idx2paths) < 50:
            scores = tuple(map(self.validate_path_pe_from_globals, idx2paths.keys()))
        else:
            with Pool(self.processors) as pool:
                scores = pool.map(self.validate_path_pe_from_globals, idx2paths.keys())
        return scores


    
    def join_paths_dispatcher(self, idx2key, toJoin, pe_cutoff, val_with_sign = False, targets = None, subsets_as_overlaps = False, assume_nosign = False, direct_join = False):
        
        if direct_join:
            goodJoins = targets
        else:
            idx2chunk = {i: chunk for i, chunk in enumerate(np.array_split(tuple(idx2key), self.processors))}
            val_seqPaths = self.signSeqPaths if val_with_sign else self.DBG.seqPaths
            val_pairings = self.signSeqPairings if val_with_sign else self.seqPairings
            self.set_multiprocessing_globals( idx2chunk, idx2key, toJoin, self.DBG, val_seqPaths, val_pairings, pe_cutoff, subsets_as_overlaps, assume_nosign, pe_cutoff )

            if self.processors == 1 or len(toJoin) < 100:
                goodJoins = [ids for idss in map(self.join_pairs, idx2chunk) for ids in idss]
            else:
                with Pool(self.processors) as pool:
                    goodJoins = [ids for idss in pool.imap(self.join_pairs, idx2chunk) for ids in idss]
            self.clear_multiprocessing_globals()

        joinedIdxs = set()
        for i, j in goodJoins:
            joinedIdxs.add(i)
            joinedIdxs.add(j)  
        singleIdx = {i for i in idx2key if i not in joinedIdxs}
        
        joinedPaths = ( overlap(toJoin[idx2key[i]], toJoin[idx2key[j]], subsets_as_overlaps = subsets_as_overlaps, assume_nosign = assume_nosign) for i, j in goodJoins )
        
        return joinedPaths, goodJoins, singleIdx
    


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
    def join_pair(cls, ids, return_merged = False):
        idx2key, toJoin, val_seqPaths, val_seqPairings, pe_cutoff, subsets_as_overlaps, assume_nosign = cls.get_multiprocessing_globals()
        i, j = ids
        k1, k2 = idx2key[i], idx2key[j]
        joinedpath = overlap(toJoin[k1], toJoin[k2], subsets_as_overlaps = subsets_as_overlaps, assume_nosign = assume_nosign)
        goodJoin = False
        if joinedpath.shape[0]:
            if not cutoff or cls.validate_path_pe(joinedpath, val_seqPaths, val_seqPairings) >= pe_cutoff:
               goodJoin = True
            elif return_merged:
                joinedpath = np.empty(0, dtype=np.uint32)
        return joinedpath if return_merged else goodJoin


    @classmethod
    def join_pairs(cls, chunk):
        idx2chunk, idx2key, toJoin, DBG, val_seqPaths, val_seqPairings, cutoff, subsets_as_overlaps, assume_nosign, pe_cutoff = cls.get_multiprocessing_globals()
        goodJoinsPre = []
        joinedPaths = []
        for i in idx2chunk[chunk]:
            cjoins = [j for j in idx2key if j > i]
            res = ( ( (i,j), overlap(toJoin[idx2key[i]], toJoin[idx2key[j]], subsets_as_overlaps = assume_nosign, assume_nosign = assume_nosign) ) for j in cjoins)
            res = [ (ids, path) for ids, path in res if path.shape[0]]
            goodJoinsPre.extend( (info[0] for info in res) )
            joinedPaths.extend( (info[1] for info in res) )
        signs2paths, goodJoins, signSeq2origin, origin2signSeq = cls.summarize_joins(joinedPaths, goodJoinsPre, DBG, val_seqPaths, val_seqPairings, pe_cutoff = pe_cutoff)
        return goodJoins
      
    
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
        idx2paths, seqPaths, seqPairings = cls.get_multiprocessing_globals()
        return cls.validate_path_pe(idx2paths[i], seqPaths, seqPairings)


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

##                if sha1(seq.encode('UTF-8')).hexdigest()[:8] == 'c4780140': #####
##                    print(v1cover)
##                    print(v2cover)
##                    print()

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
##        if sha1(seq.encode('UTF-8')).hexdigest()[:8] == 'c4780140': #####
##            print(isGood)
        return isGood
                

    @classmethod
    def validate_path_pe_competitive2(cls, i): ## still with the old pairings schema
        idx2fullSeqs, fullSeqKmers, seqKmers, seqs2names, names2seqs, pairings = cls.get_multiprocessing_globals()
        seq = idx2fullSeqs[i]
        info = fullSeqKmers[seq]
        v1cover = cls.validate_path_pe(fullSeqKmers[seq]['varray'], seqKmers, seqs2names, names2seqs, pairings, return_vertices = True)
        isGood = True
        for seq2, info2 in fullSeqKmers.items():
            if seq == seq2:
                continue
            ol = info['vsignset'] & info2['vsignset']
            if ol and len(info['vsignset']) - len(ol) < 10:
                v2cover = cls.validate_path_pe(fullSeqKmers[seq2]['varray'], seqKmers, seqs2names, names2seqs, pairings, return_vertices = True)
                if sum([v1cover[v] == 1 for v in ol]) < sum([v2cover[v] == 1 for v in ol]):
                    isGood = False
                    break
        return isGood


    @classmethod
    def extend_vertex(cls, v):
        fullSeqPaths, incompletes, DBG, maxDepth, direction = cls.get_multiprocessing_globals()
        # first search for extensions in the paths we already calculated, use the graph only if we find nothing
        tgtVertices = DBG.sources if direction == 'left' else DBG.sinks
        tgtIdx = 0 if direction == 'left' else -1

        extensions = set()
        for source in (fullSeqPaths, incompletes, DBG.seqPaths):
            for path in source.values():
                if path[tgtIdx] in tgtVertices:
                    pos = getIdx(path, v)
                    if pos < maxsize: # getIdx uses PY_SSIZE_T_MAX (sys.maxsize in python) to denote v not being present in path
                        if direction == 'left':
                            extensions.add(tuple(path[:pos]))
                        else:
                            extensions.add(tuple(path[pos+1:]))
##        if not extensions:
##            for tgt in tgtVertices:
##                s, t = (tgt, v) if direction == 'left' else (v, tgt)
##                if gt.topology.shortest_path(DBG.G, s, t, dag=True)[0]:
##                    #exts = gt.topology.all_paths(DBG.G, s, t, cutoff=maxDepth)
##                    exts = [np.array(ext, dtype=np.uint32) for ext in gt.topology.all_shortest_paths(DBG.G, s, t, dag=True)]
##                    if direction == 'left':
##                        exts = [ext[:-1] for ext in exts]
##                    else:
##                        exts = [ext[1:]  for ext in exts]
##                    extensions.extend( [ext for ext in exts if cls.validate_path_se(ext, DBG.seqKmers)] )
            
        return extensions




    @classmethod
    def validate_extension(cls, i):
        idx2candidates, idx2extensionVertices, vertex2paths = cls.get_multiprocessing_globals()
        path = idx2candidates[i]
        leftExtVertex, rightExtVertex = idx2extensionVertices[i]
        validationSeqPathsLeft = {i: path for i, path in enumerate(vertex2paths[leftExtVertex])}
        validationSeqPathsRight = {i: path for i, path in enumerate(vertex2paths[rightExtVertex])}
        r1 = cls.validate_path_se(path, validationSeqPathsLeft, allow_partial = True)
        r2 = cls.validate_path_se(path, validationSeqPathsRight, allow_partial = True)
        return r1 and r2





