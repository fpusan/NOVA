from collections import defaultdict
import numpy as np

import graph_tool as gt
from graph_tool.all import Graph

class DBG:
    def __init__(self, seqs, ksize):
        # Get kmers from sequences
        G = Graph()
        self.ksize = ksize
        seqKmers = {}
        kmer2vertex = {}
        edgeAbund = defaultdict(int)
        self.kmerAbund = defaultdict(int)
        for seq in seqs:
            if len(seq) > self.ksize:
                if seq not in seqKmers:
                    kmers = self.seq2kmers(seq)
                    fundKmers = [kmer for i, kmer in enumerate(kmers) if not i%(self.ksize-3)] # minimum set of kmers that covers the whole sequence
                    if fundKmers[-1] != kmers[-1]:                                             # we don't use the strict minimum bc the strict minimum would be a subset of a larger sequence with an insertion in that region 
                        fundKmers.append(kmers[-1])
                    seqKmers[seq] = {'set': set(kmers), 'list': kmers, 'fundset': set(fundKmers), 'fundlist': fundKmers, 'abund': 1}
                else:
                    seqKmers[seq]['abund'] += 1
                    kmers = seqKmers[seq]['list']
                for kmer in kmers:
                    self.kmerAbund[kmer] += 1
                for i in range(len(kmers)-1):
                    k1, k2 = kmers[i], kmers[i+1]
                    for km in (k1, k2):
                        if km not in kmer2vertex:
                            kmer2vertex[km] = int(G.add_vertex())
                    if (k1,k2) not in edgeAbund:
                        G.add_edge(kmer2vertex[k1], kmer2vertex[k2])
                    edgeAbund[(k1,k2)] += 1

        assert gt.topology.is_DAG(G)
        self.sources = {int(v) for v in G.vertices() if not list(v.in_neighbors()) } # We use integers rather than vertex objects bc vertex objects can't be pickled
        self.sinks   = {int(v) for v in G.vertices() if not list(v.out_neighbors())}
        self.splitters = {int(v) for v in G.vertices() if len(list(v.in_neighbors())) < 2 and len(list(v.out_neighbors())) > 1}
        self.mergers = {int(v) for v in G.vertices() if len(list(v.in_neighbors())) > 1 and len(list(v.out_neighbors())) < 2}
        self.extenders = {int(v) for v in G.vertices() if len(list(v.in_neighbors())) < 2 and len(list(v.out_neighbors())) < 2}
        self.pivots = {int(v) for v in G.vertices() if len(list(v.in_neighbors())) > 1 and len(list(v.out_neighbors())) > 1}
        self.signature = set()
        for v in self.splitters:
            self.signature.add(v)
            self.signature.update(set(G.get_out_neighbors(v)))
        for v in self.mergers:
            self.signature.add(v)
            self.signature.update(set(G.get_in_neighbors(v)))
        for v in self.pivots:
            self.signature.add(v)
            self.signature.update(set(G.get_out_neighbors(v)))
            self.signature.update(set(G.get_in_neighbors(v)))

        self.nvertices = len(list(G.vertices()))
        self.nedges = len(list(G.edges()))
        self.nsources = len(self.sources)
        self.nsinks = len(self.sinks)
        self.nsplitters = len(self.splitters)
        self.nmergers = len(self.mergers)
        self.nextenders = len(self.extenders)
        self.npivots = len(self.pivots)
        self.nsignature = len(self.signature)
        self.G = G
        self.seqKmers = seqKmers
        self.kmer2vertex = kmer2vertex
        self.vertex2kmer = {v: k for k, v in kmer2vertex.items()}

        for seq in seqKmers:
            seqKmers[seq]['varray'] = np.array([kmer2vertex[k] for k in seqKmers[seq]['list']])
            seqKmers[seq]['vset'] = set(seqKmers[seq]['varray'])
            seqKmers[seq]['vfundset'] = set([kmer2vertex[k] for k in seqKmers[seq]['fundset']])
            seqKmers[seq]['vsignset'] = seqKmers[seq]['vset'] & self.signature


    def seq2kmers(self, seq):
        return [seq[i:i+self.ksize] for i in range(len(seq)-self.ksize+1)]


    def seq2varray(self, seq): # will fail if the sequence introduces new kmers
        kmers = self.seq2kmers(seq)
        return np.array([self.kmer2vertex[k] for k in kmers])

        

    def reconstruct_sequence(self, path):
        kmers = [self.vertex2kmer[vertex] for vertex in path]
        fullseq = [kmers[0]]
        extra = [kmer[-1] for kmer in kmers[1:]]
        fullseq.extend(extra)
        fundKmers = [kmer for i, kmer in enumerate(kmers) if not i%(self.ksize-3)] # minimum set of kmers that covers the whole sequence
        if fundKmers[-1] != kmers[-1]:                                             # we don't use the strict minimum bc the strict minimum would be a subset of a larger sequence with an insertion in that region 
            fundKmers.append(kmers[-1])
        fundset = set(fundKmers)
        vset = set(path)
        vfundset = set([self.kmer2vertex[k] for k in fundset])
        vsignset = vset & self.signature

        return ''.join(fullseq), {'set': set(kmers), 'list': kmers, 'fundlist': fundKmers, 'fundset': fundset, 'varray': path, 'vset': vset, 'vfundset': vfundset, 'vsignset': vsignset}


        


