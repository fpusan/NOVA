from collections import defaultdict, namedtuple
from hashlib import sha1
import numpy as np

import graph_tool as gt
from graph_tool.all import Graph


class DBG:
    def __init__(self, seqs, ksize, min_edge_cov):
        # Get kmers from sequences
        self.ksize = ksize
        seqKmers = {}
        self.seqAbund = defaultdict(int)
        self.edgeAbund = defaultdict(int)
        self.includedSeqs = 0
        includedKmers = set()
        self.nSS = 0
        for seq in seqs:
            if len(seq) <= self.ksize:
                self.nSS += 1
            else:
                self.includedSeqs += 1
                self.seqAbund[seq] += 1
                kmers = self.seq2kmers(seq)
                includedKmers.update(kmers)
                seqKmers[seq] = kmers
                for i in range(len(kmers)-1):
                    self.edgeAbund[(kmers[i],kmers[i+1])] += 1

        # Remove low abundance edges
##        cutoff = 0.00001
##        badEdges = {edge for edge, abund in self.edgeAbund.items() if abund / self.includedSeqs <= cutoff}
        cutoff = min_edge_cov
        badEdges = {edge for edge, abund in self.edgeAbund.items() if abund <= cutoff}
        goodEdges = {edge for edge in self.edgeAbund if edge not in badEdges}
        badSeqs = set()
        for seq, kmers in seqKmers.items():
            for i in range(len(kmers)-1):
                if (kmers[i],kmers[i+1]) in badEdges:
                    badSeqs.add(seq)
                    break
        goodKmers = {kmer for seq, kmers in seqKmers.items() for kmer in kmers if seq not in badSeqs}
        badKmers = {kmer for kmer in includedKmers if kmer not in goodKmers}
        seqKmers = {seq: kmers for seq, kmers in seqKmers.items() if seq not in badSeqs}
        self.seqAbund = {seq: abund for seq, abund in self.seqAbund.items() if seq not in badSeqs}
        self.includedSeqs = sum(self.seqAbund.values())
        self.nBS, self.nBV, self.nBE = len(badSeqs), len(badKmers), len(badEdges)

        # Create DBG
        G = Graph()
        self.kmerAbund = defaultdict(int)
        self.edgeAbund = defaultdict(int)
        self.kmer2vertex = {}
        self.seqPaths = {}
        for seq, kmers in seqKmers.items():
            abund = self.seqAbund[seq]
            for k in kmers:
                self.kmerAbund[k] += abund
                if k not in self.kmer2vertex:
                    self.kmer2vertex[k] = int(G.add_vertex())
            self.seqPaths[seq] = np.array([self.kmer2vertex[kmer] for kmer in kmers], dtype=np.uint32)
            for i in range(len(kmers)-1):
                k1, k2 = kmers[i],kmers[i+1]
                edge = (k1, k2)
                if edge not in self.edgeAbund:
                    G.add_edge(self.kmer2vertex[k1], self.kmer2vertex[k2])
                self.edgeAbund[edge] += abund
        assert gt.topology.is_DAG(G)

        
        # Rename nodes so that node ids are topologically sorted
        topoIdx = {v: i for i,v in enumerate(gt.topology.topological_sort(G))}
        self.kmer2vertex = {kmer: topoIdx[v] for kmer, v in self.kmer2vertex.items()}
        G.clear_edges() # we use the same nodes, but add new edges
        for k1, k2 in self.edgeAbund:
            G.add_edge(self.kmer2vertex[k1], self.kmer2vertex[k2])
        for seq, path in self.seqPaths.items():
            self.seqPaths[seq] = np.array([topoIdx[v] for v in path], dtype=np.uint32)


        # Get signature vertices
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
        self.signature.update(self.sources)
        self.signature.update(self.sinks)
        self.signature = frozenset(self.signature)
        
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
        self.vertex2kmer = {v: k for k, v in self.kmer2vertex.items()}

        self.vertex2paths = defaultdict(list)
        for path in self.seqPaths.values():
            for v in path:
                self.vertex2paths[v].append(path)



    def seq2kmers(self, seq):
        return [seq[i:i+self.ksize] for i in range(len(seq)-self.ksize+1)]


    def seq2varray(self, seq): # will fail if the sequence introduces new kmers
        kmers = self.seq2kmers(seq)
        try:
            return np.array([self.kmer2vertex[k] for k in kmers], dtype=np.uint32)
        except:
            return np.array([], dtype=np.uint32)


    def reconstruct_sequence(self, path):
        kmers = [self.vertex2kmer[vertex] for vertex in path]
        fullseq = [kmers[0]]
        extra = [kmer[-1] for kmer in kmers[1:]]
        fullseq.extend(extra)
        return ''.join(fullseq)   
        

    def path2info(self, path, return_sequence = False):
       
        vset = frozenset(path)
        vsignset = vset & self.signature

        if return_sequence:
            return self.reconstruct_sequence(path), Path(path, vset, vsignset)
        else:
            return self.get_hash(path), Path(path, vset, vsignset)

    def write_seqs(self, paths, output_name, attributes=None, attribute_name=None):
        with open(output_name, 'w') as outfile:
            for i, path in enumerate(paths):
                seq = self.reconstruct_sequence(path)
                seqName = self.get_hash(path)
                header = f'>{seqName}'
                if attributes and attribute_name:
                    header += f'_{attribute_name}={attributes[i]}'
                outfile.write(f'{header}\n{seq}\n')
        

    @staticmethod ######## THIS SHOULD BE REMOVED
    def get_hash_string(seq):
        return sha1(seq.encode('UTF-8')).hexdigest()[:8]

    @staticmethod
    def get_hash(path):
        return sha1(path).hexdigest()
        

