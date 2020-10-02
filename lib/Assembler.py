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

    
    def __init__(self, seqData, pe_support_threshold, processors, sample, taxon, output_dir):
        self.names2seqs = seqData.sequences
        self.pairings = seqData.pairings
        self.pe_support_threshold = pe_support_threshold
        self.processors = processors
        self.sample = sample
        self.taxon = taxon
        self.output_dir = output_dir
        

    def run(self, ksize):

        if len(self.names2seqs) < 100:
            print(f'{self.sample}; {self.taxon}\tLess than 100 sequences, ignoring')
            return {}, 0


        # Map sequences to sequence names
        self.seqs2names = defaultdict(set)
        for name, seq in self.names2seqs.items():
           self.seqs2names[seq].add(name)


        # Create DBG
        print(f'{self.sample}; {self.taxon}\tCreating DBG of k={ksize} from {len(self.names2seqs)} sequences')
        DBG1 = DBG(list(self.names2seqs.values()), ksize)
        print(f'{self.sample}; {self.taxon}\t{DBG1.includedSeqs} out of {len(self.names2seqs)} reads ({round(100*DBG1.includedSeqs/len(self.names2seqs), 2)}%) were included in the graph')
        print(f'{self.sample}; {self.taxon}\t{DBG1.nvertices} vertices, {DBG1.nedges} edges, {DBG1.nsources} sources, {DBG1.nsinks} sinks, {DBG1.nsplitters} splitters, {DBG1.nmergers} joiners, {DBG1.npivots} pivots, {DBG1.nsignature} signature ({round(100*DBG1.nsignature/DBG1.nvertices,2)}%)')


        # Get seed seqs joining paired end reads
        pairs2assemble = {}
        singletons = {}
        i = 0
        for p1, p2 in self.pairings.items():
            if p2:
                s1, s2 = self.names2seqs[p1], self.names2seqs[p2]
                if s1 in DBG1.seqKmers and s2 in DBG1.seqKmers: # They may be too short
                    v1, v2 = DBG1.seqKmers[s1]['varray'], DBG1.seqKmers[s2]['varray']
                    pairs2assemble[i] = (v1, v2)
                    i += 1
                elif s1 in DBG1.seqKmers:
                    singletons[s1] = DBG1.seqKmers[s1]
                elif s2 in DBG1.seqKmers:
                    singletons[s2] = DBG1.seqKmers[s2]
            else:
                s1 = self.names2seqs[p1]
                if s1 in DBG1.seqKmers:
                    singletons[s1] = DBG1.seqKmers[s1]
                
        if not pairs2assemble:
            print(f'{self.sample}; {self.taxon}\tFound no valid pairs for assembly, ignoring') # We could still pull through if there are no (or few) signature kmers (just get paths from source to sink)
            return {}, 0


        print(f'{self.sample}; {self.taxon}\tFinding seed paths from {len(pairs2assemble)} pairs and {len(singletons)} singletons')
        MAX_DEPTH = 400 ### THIS IS IMPORTANT. IF IT'S TOO LOW WE'LL LOSE VALID PAIRS
        self.set_multiprocessing_globals(  DBG1.G, pairs2assemble, DBG1.seqKmers, ksize, MAX_DEPTH )
        if self.processors == 1:
            seedPaths = [path for paths in map(self.get_paths, pairs2assemble.keys()) for path in paths]
        else:
            with Pool(self.processors) as pool:
                seedPaths = [path for paths in pool.map(self.get_paths, pairs2assemble.keys(), chunksize=10) for path in paths]
        self.clear_multiprocessing_globals()
        
        seedSeqsKmers = dict(map(DBG1.reconstruct_sequence, seedPaths))
        seedSeqsKmers.update(singletons)


        signSeqsKmers = self.get_signature_sequences(seedSeqsKmers)
        
        singletons    = {seq: info for seq, info in signSeqsKmers.items() if seq     in singletons} # keep the singletons containing extra signKmers
        signSeqsKmers = {seq: info for seq, info in signSeqsKmers.items() if seq not in singletons} # store singletons separately

        if not signSeqsKmers:
            print(f'{self.sample}; {self.taxon}\tFound no seed paths, ignoring')
            return {}, 0


        if self.output_dir:
            self.write_seqs(seedSeqsKmers, f'{self.output_dir}/{self.taxon}/{self.sample}.{self.taxon}.seeds.fasta')
            self.write_seqs(signSeqsKmers, f'{self.output_dir}/{self.taxon}/{self.sample}.{self.taxon}.signs.fasta')
        
        kmers = {v for info in seedSeqsKmers.values() for v in info['varray']}
        percent = round(100*len(kmers & set(DBG1.kmer2vertex.values())) / len(DBG1.kmer2vertex),2)
        percentSign = round(100*len(signSeqsKmers)/len(seedSeqsKmers), 2)
        print(f'{self.sample}; {self.taxon}\t{percent}% of the original kmers are represented in {len(seedSeqsKmers)} seed paths, out of which {len(signSeqsKmers)} ({percentSign}%) are signature paths')

        step = 0
        finalJoinedSeqKmers = {}
        toJoin = signSeqsKmers
        addedSingletons = False

        # Sequentially join seed paths
        while True:
            if not toJoin:
                break
            step += 1
            cutoff = 0.5
            msg = f'{self.sample}; {self.taxon}\tSTEP {step}: Joining {len(toJoin)} candidate paths'
            if addedSingletons:
                msg += ' (included singletons)'
            else:
                msg += ' (paired-end only)'
            print(msg)
            idx2signSeq = {i: seq for i, seq in enumerate(toJoin)}
            nchunks = self.processors * 8
            self.set_multiprocessing_globals( idx2signSeq, toJoin, DBG1.seqKmers, self.seqs2names, self.names2seqs, self.pairings, ksize, cutoff, nchunks )
            if self.processors == 1 or len(toJoin) < 100:
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
                if not addedSingletons:
                    toJoin.update(singletons)
                    addedSingletons = True
                    continue
                else:
                    break

            if self.output_dir:
                self.write_seqs(signJoinedSeqKmers, f'{self.output_dir}/{self.taxon}/{self.sample}.{self.taxon}.step{step}.fasta')


            # If we couldn't join them in this step, assume we'll never can.
            for i, seq in idx2signSeq.items():
                if i not in joinedIdxs:
                    finalJoinedSeqKmers[seq] = toJoin[seq]

            toJoin = {}

            # If we joined them and they are not full, try to extend them in the next step
            for seq, info in signJoinedSeqKmers.items():
                if info['varray'][0] in DBG1.sources and info['varray'][-1] in DBG1.sinks:
                    finalJoinedSeqKmers[seq] = info
                else:
                    toJoin[seq] = info
                    

        # After we finish joining, check whether they are complete
        finalSignJoinedSeqKmers = self.get_signature_sequences(finalJoinedSeqKmers)
        fullSeqKmers = {}
        incompletes = {}
        
        for seq, info in finalSignJoinedSeqKmers.items():
            if info['varray'][0] in DBG1.sources and info['varray'][-1] in DBG1.sinks:
                fullSeqKmers[seq] = info
            else:
                incompletes[seq] = info
        incompletes = {seq: info for seq, info in incompletes.items() if len(seq) > 300}

        avgLen = round(np.mean([len(seq) for seq in incompletes]), 2)

        print(f'{self.sample}; {self.taxon}\tFound {len(fullSeqKmers)} complete paths and {len(incompletes)} incomplete paths (avg. len {avgLen})')

        if self.output_dir:
            self.write_seqs(incompletes, f'{self.output_dir}/{self.taxon}/{self.sample}.{self.taxon}.incompletes.fasta')

        leftSignVertices  = tuple({info['vsignarray'][0]  for info in incompletes.values() if info['vsignarray'][0]  not in DBG1.sources}) # Cast to tuple to ensure that order is preserved
        rightSignVertices = tuple({info['vsignarray'][-1] for info in incompletes.values() if info['vsignarray'][-1] not in DBG1.sinks  })

        # Extend sigKmers appearing at the start or end of incomplete sequences to sources and sinks
        print(f'{self.sample}; {self.taxon}\tCalculating {len(leftSignVertices)} left and {len(rightSignVertices)} right extensions')

        self.set_multiprocessing_globals( fullSeqKmers, incompletes, DBG1, ksize, 1600, 'left' ) # Larger depth cutoff here bc we might want to extend long partial sequences that were not assembled
        if self.processors == 1 or len(leftSignVertices) < 50:
            leftExtensions = dict(zip(leftSignVertices, map(self.extend_vertex, leftSignVertices)))
        else:
            with Pool(self.processors) as pool:
                leftExtensions = dict(zip(leftSignVertices, pool.map(self.extend_vertex, leftSignVertices)))
        self.clear_multiprocessing_globals()

        self.set_multiprocessing_globals( fullSeqKmers, incompletes, DBG1, ksize, 1600, 'right' ) # Larger depth cutoff here bc we might want to extend long partial sequences that were not assembled
        if self.processors == 1 or len(rightSignVertices) < 50:
            rightExtensions = dict(zip(rightSignVertices, map(self.extend_vertex, rightSignVertices)))
        else:
            with Pool(self.processors) as pool:
                rightExtensions = dict(zip(rightSignVertices, pool.map(self.extend_vertex, rightSignVertices)))
        self.clear_multiprocessing_globals()

        print(sum(map(len, leftExtensions.values())), sum(map(len, rightExtensions.values()))) ############


        # Extend incomplete paths
        print(f'{self.sample}; {self.taxon}\tExtending incomplete paths')
        i = 0
        idx2candidates = {}
        idx2extensionVertices = {}
        
        for seq, info in incompletes.items():
            
            leftSignVertex  = info['vsignarray'][0]
            rightSignVertex = info['vsignarray'][-1]
            leftSignPos  = np.where(info['varray']==leftSignVertex )[0][0] # Assuming there can't be repeated elements
            rightSignPos = np.where(info['varray']==rightSignVertex)[0][0]
            partials     = []
            fullPaths    = []
            
            if rightSignVertex not in DBG1.sinks: # Do right before so as to not mess up indexing
                for ext in rightExtensions[rightSignVertex]:
                    partials.append(np.concatenate( (info['varray'][:rightSignPos+1], ext) ))
            else:
                partials.append(info['varray'])
            
            if leftSignVertex not in DBG1.sources:
                for partial in partials:
                    for ext in leftExtensions[leftSignVertex]:
                        fullPaths.append(np.concatenate( (ext, partial[leftSignPos:]) ))
            else:
                fullPaths = partials

            #assert fullPaths # Actually it might be that we found no valid extension for a particular left/right sign vertex...

            for path in fullPaths:
                assert path[0] in DBG1.sources and path[-1] in DBG1.sinks
                idx2candidates[i] = path
                idx2extensionVertices[i] = (leftSignVertex, rightSignVertex)
                i+=1


        # Filter out extended candidates
        self.set_multiprocessing_globals( idx2candidates, idx2extensionVertices, DBG1, ksize )        
        if self.processors == 1 or len(idx2candidates) < 50:
            fullPaths = [path for path, isValid in zip(idx2candidates.values(), map(self.validate_extension, idx2candidates.keys())) if isValid]
        else:
            with Pool(self.processors) as pool:
                fullPaths = [path for path, isValid in zip(idx2candidates.values(), pool.map(self.validate_extension, idx2candidates.keys())) if isValid]
        self.clear_multiprocessing_globals()

        fullSeqKmers.update( (DBG1.reconstruct_sequence(path) for path in fullPaths) )
                

        # Calculate PE scores for candidates
        print(f'{self.sample}; {self.taxon}\tFiltering {len(fullSeqKmers)} complete sequences')
        idx2fullPaths = {i: info['varray'] for i, info in enumerate(fullSeqKmers.values())}
        self.set_multiprocessing_globals( idx2fullPaths, DBG1.seqKmers, ksize, self.seqs2names, self.names2seqs, self.pairings )
        if self.processors == 1 or len(idx2fullPaths) < 50:
            scores = map(self.validate_path_pe_from_globals, range(len(idx2fullPaths)))
        else:
            with Pool(self.processors) as pool:
                scores = pool.map(self.validate_path_pe_from_globals, range(len(idx2fullPaths)))
        
        for info, score in zip(fullSeqKmers.values(), scores): # Again trusting that iteration order is conserved in dict.values()
            info['score'] = score
        self.clear_multiprocessing_globals()
  
        if self.output_dir:
            self.write_seqs(fullSeqKmers, f'{self.output_dir}/{self.taxon}/{self.sample}.{self.taxon}.candidates.fasta', header_field = 'score')

        passingCandidates = {seq: info for seq, info in fullSeqKmers.items() if info['score'] >= 0.98} # A final filter

        if not passingCandidates:
            print(f'{self.sample}; {self.taxon}\tNo candidates with perfect score, chilling out...')
            passingCandidates = {seq: info for seq, info in fullSeqKmers.items() if info['score'] >= 0.7} # A final filter, more lenient

        if not passingCandidates:
             print(f'{self.sample}; {self.taxon}\tNo valid candidates were found, ignoring')
             return {}, 0

        fullSeqKmers = passingCandidates

        print(f'{self.sample}; {self.taxon}\tRunning competitive filter over {len(fullSeqKmers)} complete sequences')
        idx2fullSeqs = {i: seq for i, seq in enumerate(fullSeqKmers)}
        self.set_multiprocessing_globals( idx2fullSeqs, fullSeqKmers, DBG1.seqKmers, ksize, self.seqs2names, self.names2seqs, self.pairings )
        if self.processors == 1 or len(fullSeqKmers) < 50:
            competitiveFilter = map(self.validate_path_pe_competitive, range(len(idx2fullSeqs)))
        else:
            with Pool(self.processors) as pool:
                competitiveFilter = pool.map(self.validate_path_pe_competitive, range(len(idx2fullSeqs)))
        fullSeqKmers = dict([(seq, info) for (seq, info), isGood in zip(fullSeqKmers.items(), competitiveFilter) if isGood])
        self.clear_multiprocessing_globals()


        # Build equation system
        print(f'{self.sample}; {self.taxon}\tEstimating the abundance of {len(fullSeqKmers)} candidate sequences')
        x = []
        y = []

        for kmer, abund in DBG1.kmerAbund.items():
            vertex = DBG1.kmer2vertex[kmer]
            y.append(abund)
            eq = []
            for fs, fs_kmers in fullSeqKmers.items(): # assuming iteration order in dicts is stable which should be above 3.7
                if vertex in fs_kmers['vset']:
                    eq.append(1)
                else:
                    eq.append(0)
            x.append(eq)


        # Go for the eyes, boo!
        abunds, residual = nnls(x,y)

        for abund, info in zip(abunds, fullSeqKmers.values()):
            info['abund'] = abund

        if self.output_dir:
            self.write_seqs(fullSeqKmers, f'{self.output_dir}/{self.taxon}/{self.sample}.{self.taxon}.final.fasta', header_field = 'abund')


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
                if info['vset'].issubset(fsinfo['vset']): ### CAMBIAR A VFUND SETS!
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
                key = tuple(info['vsignarray'])
                if len(seq) > len(signSeqs[key]): # For the same set of signature kmers, consider only the longest sequence
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
        """
        Return whether a sequence is a subset (based on signature kmers) of a second, longer one.
        """
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
        G, pairs2assemble, seqKmers, ksize, maxDepth = cls.get_multiprocessing_globals()
        source, sink = pairs2assemble[i]
        leading = source[:-1]
        trailing = sink[1:]
        paths = [path for path in gt.topology.all_paths(G, source[-1], sink[0], cutoff=maxDepth)]
        paths = [np.concatenate([leading, path, trailing]) for path in paths]
        paths = [path for path in paths if cls.validate_path_se(path, seqKmers, ksize)]                       
        return paths


    @classmethod
    def join_paths_dispatcher(cls, chunkidx):
        paths = []
        joinedIdxs = set()
        idx2seedSeq, seedSeqKmers, seqKmers, seqs2names, names2seqs, pairings, ksize, cutoff, nchunks = cls.get_multiprocessing_globals()
        for i, (idx1, idx2) in enumerate(combinations(idx2seedSeq, 2)):
            if i%nchunks == chunkidx:
                joinedpath = cls.overlap(idx2seedSeq[idx1], idx2seedSeq[idx2], seedSeqKmers, ksize, return_merged = True, subseqs_as_overlaps = False)
                if len(joinedpath):
                    if not cutoff or cls.validate_path_pe(joinedpath, seqKmers, ksize, seqs2names, names2seqs, pairings) >= cutoff:
                        paths.append(joinedpath)
                        joinedIdxs.add(idx1)
                        joinedIdxs.add(idx2)
        return paths, joinedIdxs
  

    @classmethod
    def overlap(cls, s1, s2, seqKmers, ksize, s2info = None, return_merged = False, subseqs_as_overlaps = True):
        
        s1array   = seqKmers[s1]['varray']
        s1set     = seqKmers[s1]['vset']
        s1signset = seqKmers[s1]['vsignset']
        s1start   = seqKmers[s1]['varray'][0]
        s1end     = seqKmers[s1]['varray'][-1]
        
        if s2info:
            s2array   = s2info[0]
            s2set     = s2info[1]
            s2start   = s2array[0]
            s2end     = s2array[-1]
            
        else:
            s2array   = seqKmers[s2]['varray']
            s2set     = seqKmers[s2]['vset']
            s2signset = seqKmers[s2]['vsignset']
            s2start   = s2array[0]
            s2end     = s2array[-1]

        res = np.array([]) if return_merged else False

        if s1start in s2set and s1end in s2set: # s1 may be contained in s2
            if subseqs_as_overlaps and s1signset.issubset(s2set):
                res = s2array if return_merged else True
        elif s1start in s2set and s2end in s1set: # s2 starts before
            idx = np.where(s2array==s1start)[0][0] # Assuming there can't be repeated elements
            subpath = set((v for i,v in enumerate(s2array[idx:]) if not i%(ksize/4)))
            if subpath.issubset(s1set):
                res = np.concatenate([s2array[:idx], s1array]) if return_merged else True
        elif s1end in s2set and s2start in s1set: # s1 starts before
            idx = np.where(s2array==s1end)[0][0] # Assuming there can't be repeated elements
            subpath = set((v for i,v in enumerate(s2array[:idx+1]) if not i%(ksize/4))) ### This will fail for ksizes not divisible by 4!!!
            if subpath.issubset(s1set):
                res = np.concatenate([s1array, s2array[(idx+1):]]) if return_merged else True
        elif s2start in s1set and s2end in s1set: # s2 may be contained in s1
            if subseqs_as_overlaps and s2signset.issubset(s1set):
                res = s1array if return_merged else True
        else:
            pass
        return res 
        
    
    @classmethod
    def validate_path_se(cls, path, seqKmers, ksize, allow_partial = False):
        pathKmerSet     = set(path)
        seenKmers       = set()

        for seq in seqKmers:
            if cls.overlap(seq, path, seqKmers, ksize, s2info=(path, pathKmerSet) ):
                seenKmers.update(seqKmers[seq]['vset'])

        if allow_partial:
            return len(seenKmers) > 0
        else:
            return pathKmerSet.issubset(seenKmers) # there are paths in seenKmers that are not from this path, but it's faster this way


    @classmethod
    def validate_path_pe(cls, path, seqKmers, ksize, seqs2names, names2seqs, pairings, force = False, return_vertices = False):
        pathKmerSet = set(path)

        seenKmersPaired = set()
        seenKmersPairedDict = {v: 0 for v in path}
        confirmedKmersPaired = set()
        confirmedKmersPairedDict = {v: 0 for v in path}

        mappedNames = set()

        for seq in seqKmers:
            if cls.overlap(seq, path, seqKmers, ksize, s2info=(path, pathKmerSet) ):
                mappedNames.update(seqs2names[seq])

        mappedPairs = 0
        totalPairs = 0
        for pair1 in mappedNames:
            pair2 = pairings[pair1]
            if pair2:
                totalPairs += 1
                seenKmersPaired.update(seqKmers[names2seqs[pair1]]['vset'])
                if return_vertices:
                    for v in seqKmers[names2seqs[pair1]]['vset']:
                        seenKmersPairedDict[v] += 1
                if pair2 in mappedNames:
                    mappedPairs += 1
                    confirmedKmersPaired.update(seqKmers[names2seqs[pair1]]['vset'])
                    confirmedKmersPaired.update(seqKmers[names2seqs[pair2]]['vset'])
                    if return_vertices:
                        for v in seqKmers[names2seqs[pair1]]['vset']:
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
        idx2paths, seqKmers, ksize, seqs2names, names2seqs, pairings = cls.get_multiprocessing_globals()
        return cls.validate_path_pe(idx2paths[i], seqKmers, ksize, seqs2names, names2seqs, pairings)


    @classmethod
    def validate_path_pe_competitive(cls, i):
        idx2fullSeqs, fullSeqKmers, seqKmers, ksize, seqs2names, names2seqs, pairings = cls.get_multiprocessing_globals()
        seq = idx2fullSeqs[i]
        info = fullSeqKmers[seq]
        v1cover = cls.validate_path_pe(fullSeqKmers[seq]['varray'], seqKmers, ksize, seqs2names, names2seqs, pairings, return_vertices = True)
        isGood = True
        for seq2, info2 in fullSeqKmers.items():
            if seq == seq2:
                continue
            ol = info['vsignset'] & info2['vsignset']
            if ol and len(info['vsignset']) - len(ol) < 20:
                v2cover = cls.validate_path_pe(fullSeqKmers[seq2]['varray'], seqKmers, ksize, seqs2names, names2seqs, pairings, return_vertices = True)
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
                

    @classmethod
    def validate_path_pe_competitive2(cls, i):
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
        fullSeqKmers, incompletes, DBG, ksize, maxDepth, direction = cls.get_multiprocessing_globals()
        # first search for extensions in the paths we already calculated, use the graph only if we find nothing
        tgtVertices = DBG.sources if direction == 'left' else DBG.sinks
        tgtIdx = 0 if direction == 'left' else -1
        
        pathsWithExts = [info['varray'] for info in fullSeqKmers.values() if v in info['vsignset']]
        pathsWithExts.extend([info['varray'] for info in incompletes.values()  if v in info['vsignset'] and info['varray'][tgtIdx] in tgtVertices])
        pathsWithExts.extend([info['varray'] for info in DBG.seqKmers.values() if v in info['vsignset'] and info['varray'][tgtIdx] in tgtVertices])
        extensions = set()
        for path in pathsWithExts:
            if direction == 'left':
                leftPos  = np.where(path==v)[0][0] # Assuming there can't be repeated elements
                extensions.add(tuple(path[:leftPos]))
            else:
                rightPos  = np.where(path==v)[0][0] # Assuming there can't be repeated elements
                extensions.add(tuple(path[rightPos+1:]))
        extensions = [np.array(path) for path in extensions]
        if not extensions:
            for tgt in tgtVertices:
                s, t = (tgt, v) if direction == 'left' else (v, tgt)
                if gt.topology.shortest_path(DBG.G, s, t, dag=True)[0]:
                    exts = gt.topology.all_paths(DBG.G, s, t, cutoff=maxDepth)
                    if direction == 'left':
                        exts = [ext[:-1] for ext in exts]
                    else:
                        exts = [ext[1:]  for ext in exts]
                    extensions.extend( [ext for ext in exts if cls.validate_path_se(ext, DBG.seqKmers, ksize)] )
            
        return extensions


    @classmethod
    def validate_extension(cls, i):
        idx2candidates, idx2extensionVertices, DBG, ksize = cls.get_multiprocessing_globals()
        path = idx2candidates[i]
        leftSignVertex, rightSignVertex = idx2extensionVertices[i]
        validationSeqKmersLeft  = {seq: info for seq, info in DBG.seqKmers.items() if leftSignVertex  in info['vsignset']}
        validationSeqKmersRight = {seq: info for seq, info in DBG.seqKmers.items() if rightSignVertex in info['vsignset']}
        return cls.validate_path_se(path, validationSeqKmersLeft, ksize, allow_partial = True) and cls.validate_path_se(path, validationSeqKmersRight, ksize, allow_partial = True)
##        return cls.validate_path_se(path, DBG.seqKmers, ksize)


    @staticmethod
    def get_hash(seq):
        return sha1(seq.encode('UTF-8')).hexdigest()[:8]


    @staticmethod
    def write_seqs(seqKmers, path, header_field=None):
        with open(path, 'w') as outfile:
            for seq, info in seqKmers.items():
                seqName = Assembler.get_hash(seq)
                header = f">{seqName}"
                if header_field:
                    header += f"_{header_field}={info[header_field]}"
                outfile.write(f"{header}\n{seq}\n")





class Loader():
    def __init__(self, fasta):
        self.sequences = self.fasta2dict(fasta)
        pairs = defaultdict(list)
        pairings = {}
        for header in self.sequences:
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

    @staticmethod
    def fasta2dict(fasta):
        res = {}
        for seq in open(fasta).read().strip().lstrip('>').split('>'):
            name, seq = seq.split('\n',1)
            name = name.split('\t')[0]
            res[name] = seq.replace('\n','').replace('N','').replace('-','').replace('.','')
        return res

##x = Assembler(Loader('/home/fer/Projects/smiTE/Mock_Alteromonadales200NNN/16S/NOVA/Pseudoalteromonadaceae/S_0.Pseudoalteromonadaceae.align'), 0.9, 24, 'S_0', 'Pseudoalteromonadaceae', '.')
##x.run(96)

##x = Assembler(Loader('/home/fpuente/zobel/Projects/smiTE/Mock_Alteromonadales200NNN/16S/NOVA/Pseudoalteromonadaceae/S_0.Pseudoalteromonadaceae.align'), 0.9, 24, 'S_0', 'Pseudoalteromonadaceae', '.')
##x.run(96)


##if __name__ == '__main__':
##    import yappi
##    yappi.start()
##    try:
##        x = Assembler(Loader('/home/fer/Projects/smiTE/Mock_Alteromonadales200NNN/16S/NOVA/Pseudoalteromonadaceae/S_0.Pseudoalteromonadaceae.align'), 0.9, 1, 'S_0', 'Pseudoalteromonadaceae', '.')
##        x.run(96)
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
##x = Assembler(Loader('/home/fer/Projects/smiTE/Mock_Alteromonadales200NNN/16S/NOVA/Pseudoalteromonadaceae/S_0.Pseudoalteromonadaceae.align'), 0.9, 1, 'S_0', 'Pseudoalteromonadaceae', '.')##lp.add_function(x.get_paths)
###lp.add_function(x.join_paths_dispatcher)
###lp.add_function(x.validate_path_se)
###lp.add_function(x.validate_path_pe)
##lp.add_function(x.extend_vertex)
##lp.add_function(x.overlap)
##lp_wrapper = lp(x.run)
##lp_wrapper(96)
##lp.print_stats()
 
