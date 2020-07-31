from collections import defaultdict
from itertools import combinations
from math import floor, ceil
from multiprocessing import Pool
from hashlib import sha1

from lib.Graph import DBG
import graph_tool as gt
from graph_tool.all import Graph

import numpy as np
import graph_tool as gt

from scipy.optimize import nnls


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

    
    def __init__(self, align, ksize, pe_support_threshold, processors, sample, taxon, output_dir):
        self.names2seqs = align.to_seqdict(degap=True)
        self.pairings = align.pairings
        self.ksize = ksize
        self.pe_support_threshold = pe_support_threshold
        self.processors = processors
        self.sample = sample
        self.taxon = taxon
        self.output_dir = output_dir
        

    def run(self):

        if len(self.names2seqs) < 50:
            print(f'{self.sample}; {self.taxon}\tLess than 100 sequences, ignoring')
            return {}, 0

        # Map sequences to sequence names
        self.seqs2names = defaultdict(set)
        for name, seq in self.names2seqs.items():
           self.seqs2names[seq].add(name)

        print(f'{self.sample}; {self.taxon}\tCreating DBG of k={self.ksize} from {len(self.names2seqs)} sequences')
        DBG1 = DBG(list(self.names2seqs.values()), self.ksize)
        print(f'{self.sample}; {self.taxon}\t{DBG1.nvertices} vertices, {DBG1.nedges} edges, {DBG1.nsources} sources, {DBG1.nsinks} sinks, {DBG1.nsplitters} splitters, {DBG1.nmergers} joiners, {DBG1.npivots} pivots, {DBG1.nsignature} signature')

        # Get seed seqs joining paired end reads
        pairs2assemble = {}
        i = 0
        for p1, p2 in self.pairings.items():
            if p2:
                s1, s2 = self.names2seqs[p1], self.names2seqs[p2]
                if s1 in DBG1.seqKmers and s2 in DBG1.seqKmers: # They may be too short
                    v1, v2 = DBG1.seqKmers[s1]['varray'], DBG1.seqKmers[s2]['varray']
                    pairs2assemble[i] = (v1, v2)
                    i += 1

        print(f'{self.sample}; {self.taxon}\tFinding seed paths from {len(pairs2assemble)} pairs')
        MAX_DEPTH = 300 ### THIS IS IMPORTANT. IF IT'S TOO LOW WE'LL LOSE VALID PAIRS
        self.set_multiprocessing_globals(  DBG1.G, pairs2assemble, DBG1.seqKmers, MAX_DEPTH )
        if self.processors == 1:
            seedPaths = [path for paths in map(self.get_paths, pairs2assemble.keys()) for path in paths]
        else:
            with Pool(self.processors) as pool:
                seedPaths = [path for paths in pool.map(self.get_paths, pairs2assemble.keys(), chunksize=10) for path in paths]
        self.clear_multiprocessing_globals()
        
        seedSeqsKmers = dict(map(DBG1.reconstruct_sequence, seedPaths))


        signSeqsKmers = self.get_signature_sequences(seedSeqsKmers)

        if self.output_dir:
            self.write_seqs(seedSeqsKmers, f'{self.output_dir}/{self.sample}.{self.taxon}.seeds.fasta')
            self.write_seqs(signSeqsKmers, f'{self.output_dir}/{self.sample}.{self.taxon}.signs.fasta')
        
        kmers = {kmer for info in seedSeqsKmers.values() for kmer in info['list']}
        percent = round(100*len(kmers & set(DBG1.kmer2vertex)) / len(DBG1.kmer2vertex),2)

        print(f'{self.sample}; {self.taxon}\t{percent}% of the original kmers are represented in {len(seedSeqsKmers)} seed paths, out of which {len(signSeqsKmers)} are signature paths')

        step = 0
        fullSeqKmers = {}
        
        while True:
            step += 1
            cutoff = 0.75
            print(f'{self.sample}; {self.taxon}\tSTEP {step}: Joining {len(signSeqsKmers)} signature paths')
            idx2signSeq = {i: seq for i, seq in enumerate(signSeqsKmers)}
            nchunks = self.processors * 8
            self.set_multiprocessing_globals( idx2signSeq, signSeqsKmers, DBG1.seqKmers, self.seqs2names, self.names2seqs, self.pairings, cutoff, nchunks )
            if self.processors == 1 or len(signSeqsKmers) < 100:
                results = map(self.join_paths_dispatcher, range(nchunks))       
            else:
                with Pool(self.processors) as pool:
                    results = pool.map(self.join_paths_dispatcher, range(nchunks))
            joinedPaths = []
            joinedIdxs = set()
            for res in results:
                joinedPaths.extend(res[0])
                joinedIdxs.update(res[1])
            self.clear_multiprocessing_globals()
            joinedSeqKmers = dict( DBG1.reconstruct_sequence(path) for path in joinedPaths if len(path) )
            signJoinedSeqKmers = self.get_signature_sequences(joinedSeqKmers)

            if not signJoinedSeqKmers:
                break

            if self.output_dir:
                self.write_seqs(signJoinedSeqKmers, f'{self.output_dir}/{self.sample}.{self.taxon}.step{step}.fasta')

            for seq, info in signJoinedSeqKmers.items():
                if info['varray'][0] in DBG1.sources and info['varray'][-1] in DBG1.sinks:
                    fullSeqKmers[seq] = info

            incompletes = {seq: info for seq, info in signJoinedSeqKmers.items() if seq not in fullSeqKmers}
            for i in idx2signSeq:
                if i not in joinedIdxs:
                    signSeq = idx2signSeq[i]
                    incompletes[signSeq] = signSeqsKmers[signSeq]
            signSeqsKmers = incompletes


        print(f'{self.sample}; {self.taxon}\tExtending incomplete paths')

        if self.output_dir:
            self.write_seqs(signSeqsKmers, f'{self.output_dir}/{self.sample}.{self.taxon}.incompletes.fasta')

        notExtended = 0
        for seq, info in signSeqsKmers.items():
            path = info['varray']
            fullPaths = []
            left_extensions = []
            right_extensions = []
            
            if path[0] not in DBG1.sources:
                for source in DBG1.sources:
                    exts = gt.topology.all_paths(DBG1.G, source, path[0], cutoff=MAX_DEPTH)
                    left_extensions.extend( [ext[:-1] for ext in exts] ) # Last node is already included in path
            if path[-1] not in DBG1.sinks:
                for sink in DBG1.sinks:
                    exts = gt.topology.all_paths(DBG1.G, path[-1], sink, cutoff=MAX_DEPTH)
                    right_extensions.extend( [ext[1:] for ext in exts] ) # First node is already included in path

            if left_extensions and right_extensions:
                for lext in left_extensions:
                    for rext in right_extensions:
                        fullPaths.append(np.concatenate([lext, path, rext]))
            elif left_extensions:
                for lext in left_extensions:
                    fullPaths.append(np.concatenate([lext, path]))
            elif right_extensions:
                for rext in right_extensions:
                    fullPaths.append(np.concatenate([path, rext]))
            else:
                notExtended += 1
                continue
                    
            fullSeqKmers.update( [DBG1.reconstruct_sequence(path) for path in fullPaths] )

        # Calculate PE scores for candidates
        for info in fullSeqKmers.values():
            info['score'] = self.validate_path_pe(info['varray'], DBG1.seqKmers, self.seqs2names, self.names2seqs, self.pairings)
        
        if self.output_dir:
            self.write_seqs(fullSeqKmers, f'{self.output_dir}/{self.sample}.{self.taxon}.candidates.fasta', header_field = 'score')

        passingCandidates = {seq: info for seq, info in fullSeqKmers.items() if info['score'] >= 1} # A final filter

        if not passingCandidates:
            print(f'{self.sample}; {self.taxon}\tNo candidates with perfect score, chilling out...')
            passingCandidates = {seq: info for seq, info in fullSeqKmers.items() if info['score'] >= 0.9} # A final filter, more lenient

        fullSeqKmers = passingCandidates
       

        print(f'{self.sample}; {self.taxon}\tEstimating the abundance of {len(fullSeqKmers)} candidate sequences')
        # Build equation system
        x = []
        y = []

        for kmer, abund in DBG1.kmerAbund.items():
            y.append(abund)
            eq = []
            for fs, fs_kmers in fullSeqKmers.items(): # assuming iteration order in dicts is stable which should be above 3.7
                if kmer in fs_kmers['set']:
                    eq.append(1)
                else:
                    eq.append(0)
            x.append(eq)

        # Go for the eyes, boo!
        abunds, residual = nnls(x,y)

        for abund, info in zip(abunds, fullSeqKmers.values()):
            info['abund'] = abund

        if self.output_dir:
            self.write_seqs(fullSeqKmers, f'{self.output_dir}/{self.sample}.{self.taxon}.final.fasta', header_field = 'abund')

        # Transform "sequence abundances" into read counts
        fullSeqCounts = defaultdict(float)
        totalReads = 0
        unrecoveredID = f'{self.taxon}.Unrecovered'
        
        for seq, info in DBG1.seqKmers.items():
            counts = info['abund']
            totalReads += counts
            hits = set()
            for fullSeq, fsinfo in fullSeqKmers.items():
                if not fsinfo['abund']:
                    continue
                if info['set'].issubset(fsinfo['set']): ### CAMBIAR A VFUND SETS!
                    hits.add(fullSeq)
            if not hits:
                fullSeqCounts[unrecoveredID] += counts
            else:
                hitsAbund = sum([fullSeqKmers[hit]['abund'] for hit in hits])
                for hit in hits:
                    fullSeqCounts[hit] += counts * fullSeqKmers[hit]['abund'] / hitsAbund


        # Round nicely
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
        print(f'{self.sample}; {self.taxon}\t{countsInFulls} out of {totalReads} reads ({percentAssigned}%) were assigned to {nfulls} variants')

        return fullSeqCounts, residual



    def get_signature_sequences(self, seqKmers):
        signSeqs = defaultdict(str) # {tuple_of_sorted_signature_vertices: seq}
        for seq, info in seqKmers.items():
            if info['vsignset']:
                key = tuple(sorted(info['vsignset']))
                if len(seq) > len(signSeqs[key]):
                    signSeqs[key] = seq
        idx2sign = {i: sign for i, sign in enumerate(signSeqs)}
        signLengths = {sign: len(sign) for sign in signSeqs}
        signSets = {sign: set(sign) for sign in signSeqs}
        self.set_multiprocessing_globals( ( idx2sign, signSets, signLengths ) )
        if self.processors == 1 or len(signSeqs) < 100:
            subsets = {idx2sign[k] for k in [self.signOverlap(i,j) for i, j in combinations(idx2sign, 2)] if k is not None}
        else:
            with Pool(self.processors) as pool:
                subsets = {idx2sign[k] for k in pool.starmap(self.signOverlap, [(i,j) for i, j in combinations(idx2sign, 2)]) if k is not None}
        self.clear_multiprocessing_globals()
        signSeqs = {sign: seq for sign, seq in signSeqs.items() if sign not in subsets}
        signKmers = {seq: seqKmers[seq] for seq in signSeqs.values()}
        return signKmers


    @classmethod
    def signOverlap(cls, i, j):
        idx2sign, signSets, signLengths = cls.get_multiprocessing_globals()
        sign1 = idx2sign[i]
        sign2 = idx2sign[j]
        sign1s, sign2s = signSets[sign1], signSets[sign2]
        sign1l, sign2l = signLengths[sign1], signLengths[sign2]
        if sign1l < sign2l and sign1[0] in sign2s and sign1[-1] in sign2s:
            if sign1s.issubset(sign2s):
                return i
        elif sign2l < sign1l and sign2[0] in sign1s and sign2[-1] in sign1s:
            if sign2s.issubset(sign1s):
                return j
        else:
            return None        
        

    @classmethod
    def get_paths(cls, i):
        G, pairs2assemble, seqKmers, maxDepth = cls.get_multiprocessing_globals()
        source, sink = pairs2assemble[i]
        leading = source[:-1]
        trailing = sink[1:]
        paths = [path for path in gt.topology.all_paths(G, source[-1], sink[0], cutoff=maxDepth)]
        paths = [np.concatenate([leading, path, trailing]) for path in paths]
        paths = [path for path in paths if cls.validate_path_se(path, seqKmers)]                       
        return paths

    

    @classmethod
    def join_paths_dispatcher(cls, chunkidx):
        paths = []
        joinedIdxs = set()
        idx2seedSeq, seedSeqKmers, seqKmers, seqs2names, names2seqs, pairings, cutoff, nchunks = cls.get_multiprocessing_globals()
        for i, (idx1, idx2) in enumerate(combinations(idx2seedSeq, 2)):
            if i%nchunks == chunkidx:
                joinedpath = cls.overlap(idx2seedSeq[idx1], idx2seedSeq[idx2], seedSeqKmers, return_merged = True, subseqs_as_overlaps = False)
                if len(joinedpath) and cls.validate_path_pe(joinedpath, seqKmers, seqs2names, names2seqs, pairings) >= cutoff:
                    paths.append(joinedpath)
                    joinedIdxs.add(idx1)
                    joinedIdxs.add(idx2)
        return paths, joinedIdxs
  

    @classmethod
    def overlap(cls, s1, s2, seqKmers, s2info = None, return_merged = False, subseqs_as_overlaps = True):
        ksize     = len(seqKmers[s1]['list'][0]) # assuming s1 and s2 are the same ksize
        
        s1array   = seqKmers[s1]['varray']
        s1set     = seqKmers[s1]['vset']
        s1fundset = seqKmers[s1]['vfundset']
        s1start   = seqKmers[s1]['varray'][0]
        s1end     = seqKmers[s1]['varray'][-1]
        
        if s2info:
            s2array   = s2info[0]
            s2set     = s2info[1]
            s2start   = s2array[0]
            s2end     = s2array[-1]
            s2Idxs    = s2info[2]
            
        else:
            s2array   = seqKmers[s2]['varray']
            s2set     = seqKmers[s2]['vset']
            s2fundset = seqKmers[s2]['vfundset']
            s2start   = s2array[0]
            s2end     = s2array[-1]
            s2Idxs    = {v: i for i,v in enumerate(s2array)}

        res = np.array([]) if return_merged else False

        if s1start in s2set and s1end in s2set: # s1 may be contained in s2
            if subseqs_as_overlaps and s1fundset.issubset(s2set):
                res = s2array if return_merged else True
        elif s1start in s2set and s2end in s1set: # s2 starts before
            idx = s2Idxs[s1start]
            subpath = set((v for i,v in enumerate(s2array[idx:]) if not i%(ksize/4)))
            if subpath.issubset(s1set):
                res = np.concatenate([s2array[:idx], s1array]) if return_merged else True
        elif s1end in s2set and s2start in s1set: # s1 starts before
            idx = s2Idxs[s1end]
            subpath = set((v for i,v in enumerate(s2array[:idx+1]) if not i%(ksize/4))) ### This will fail for ksizes not divisible by 4!!!
            if subpath.issubset(s1set):
                res = np.concatenate([s1array, s2array[(idx+1):]]) if return_merged else True
        elif s2start in s1set and s2end in s1set: # s2 may be contained in s1
            if subseqs_as_overlaps and s2fundset.issubset(s1set):
                res = s1array if return_merged else True
        else:
            pass
        return res 
        
    
    @classmethod
    def validate_path_se(cls, path, seqKmers):
        pathKmerSet     = set(path)
        pathKmerIndex   = {v: i for i,v in enumerate(path)}
        seenKmers       = set()

        for seq in seqKmers:
            if cls.overlap(seq, path, seqKmers, s2info=(path, pathKmerSet, pathKmerIndex) ):
                seenKmers.update(seqKmers[seq]['vset'])
        
        return pathKmerSet.issubset(seenKmers) # there are paths in seenKmers that are not from this path, but it's faster this way


    @classmethod
    def validate_path_pe(cls, path, seqKmers, seqs2names, names2seqs, pairings):
        pathKmerSet = set(path)
        pathKmerIndex = {v: i for i,v in enumerate(path)}

        seenKmersPaired = set()
        confirmedKmersPaired= set()

        mappedNames = set()

        for seq in seqKmers:
            if cls.overlap(seq, path, seqKmers, s2info=(path, pathKmerSet, pathKmerIndex) ):
                mappedNames.update(seqs2names[seq])

        mappedPairs = 0
        totalPairs = 0
        for pair1 in mappedNames:
            pair2 = pairings[pair1]
            if pair2:
                totalPairs += 1
                seenKmersPaired.update(seqKmers[names2seqs[pair1]]['vset'])
                if pair2 in mappedNames:
                    mappedPairs += 1
                    confirmedKmersPaired.update(seqKmers[names2seqs[pair1]]['vset'])
                    confirmedKmersPaired.update(seqKmers[names2seqs[pair2]]['vset'])
        assert totalPairs
        return len(confirmedKmersPaired & seenKmersPaired) / len(seenKmersPaired)


    @staticmethod
    def write_seqs(seqKmers, path, header_field = None):
        with open(path, 'w') as outfile:
            for seq, info in seqKmers.items():
                seqName = sha1(seq.encode('UTF-8')).hexdigest()[:8]
                header = f'>{seqName}'
                if header_field:
                    header += f'_{header_field}={info[header_field]}'
                outfile.write(f'{header}\n{seq}\n')



class Loader():
    def __init__(self, fasta):
        self.seqdict = self.fasta2dict(fasta)
        pairs = defaultdict(list)
        pairings = {}
        for header in self.seqdict:
            pair = header.split('_pair_')[0]
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

    def to_seqdict(self, degap=True):
        if not degap:
            res = self.seqdict
        else:
            res = {name: seq.replace('-','').replace('.','') for name, seq in self.seqdict.items()}
        return res

    @staticmethod
    def fasta2dict(fasta):
        res = {}
        for seq in open(fasta).read().strip().lstrip('>').split('>'):
            name, seq = seq.split('\n',1)
            name = name.split('\t')[0]
            res[name] = seq.replace('\n','')#.replace('N','')
        return res


##x = Assembler(Loader('/home/fer/Projects/smiTE/mock1/16S/NOVAtest/S_0.Neisseriaceae.align'), 64 , 0.9, 18, 'S_0', 'Ruminococcaceae', '/home/fer/Projects/smiTE/mock1/checkDB/NOVAtest', None)
##x.run()
##
##if __name__ == '__main__':
##    import yappi
##    yappi.start()
##    try:
##        x = Assembler(Loader('/home/fer/Projects/smiTE/mock1/16S/NOVAtest/S_0.Neisseriaceae.align'), 16 , 0.9, 12, 'S_0', 'Ruminococcaceae', None, None)
##        x.run()
##    finally:
##        func_stats = yappi.get_func_stats()
##        func_stats.save('callgrind.out', 'CALLGRIND')
##
##        yappi.stop()


#mothur "#align.seqs(fasta=candidates.fasta, reference=Oral3.mock1.S_0.16S.align)" && wc -l candidates.align.report && grep -c 100.00 candidates.align.report  && rm *logfile
#mothur "#align.seqs(fasta=final.fasta, reference=Oral3.mock1.S_0.16S.align)" && wc -l final.align.report && grep -c 100.00 final.align.report  && rm *logfile
    

##from line_profiler import LineProfiler
##
##lp = LineProfiler()
##x = Assembler(Loader('/home/fer/Projects/smiTE/mock1/16S/NOVAtest/S_0.Ruminococcaceae.align'), 18, 0.9, 1, 'S_0', 'Ruminococcaceae', None, None)
##lp.add_function(x.get_paths)
##lp.add_function(x.join_paths_dispatcher)
##lp.add_function(x.validate_path_se)
##lp.add_function(x.validate_path_pe)
##lp_wrapper = lp(x.run)
##lp_wrapper()
##lp.print_stats()






