```

gtcomplex 1.0.6 (compiled with GPU support)

GTcomplex, HPC biomolecular complex structure alignment, superposition and search.
(C)2021-2025 Mindaugas Margelevicius, Institute of Biotechnology, Vilnius University


Usage (one of the two):
gtcomplex --qrs=(<structs>|<dirs>|<archs>) --rfs=(<structs>|<dirs>|<archs>) -o <out_dir> [<options>]
gtcomplex --cls=(<structs>|<dirs>|<archs>) -o <out_dir> [<options>]   *GPU version*

Basic options:
--qrs=(<structs>,<dirs>,<archs>)
                            Comma-separated list of structure files (PDB,
                            PDBx/mmCIF, and gzip), tar archives (of structure
                            files gzipped or not) and/or directories of
                            query structures. If a directory is specified,
                            subdirectories up to 3 levels deep will be
                            searched for structures.
--rfs=(<structs>,<dirs>,<archs>)
                            Comma-separated list of structure files (PDB,
                            PDBx/mmCIF, and gzip), tar archives and/or
                            directories of reference structures (to align
                            queries with). For directories, subdirectories
                            up to 3 levels deep will be searched.
                            RECOMMENDED: -c <dir> when --speed > 9.
--sfx=<file_extension_list> Comma-separated list of extensions of structures
                            to be searched for in the directories/archives
                            specified by --qrs and --rfs (or --cls).
                            By default, all extensions are valid.
-o <output_directory>       Directory of output files for each query or
                            cluster.
-c <cache_directory>        Directory for cached data, which can provide a
                            considerable speedup for multiple queries or
                            clustering and can be reused later (for same
                            --rfs or --cls). By default, not used.
--chains                    Instead of aligning complexes, align their
                            individual chains independently.

Clustering options: (NOTE: Clustering of complexes not implemented yet)
--cls=(<structs>,<dirs>,<archs>)
                            Comma-separated list of structure files (PDB,
                            PDBx/mmCIF, and gzip), tar archives (of files
                            gzipped and not) and directories (see --qrs) of
                            structures to be clustered.
                            NOTE: The clustering criterion defined by --sort.
                            RECOMMENDED: --speed=13 for large datasets.
                            RECOMMENDED: -c <dir> when --speed > 9.
--cls-threshold=<threshold> TM-score (equal or greater) or RMSD (equal or
                            less) threshold for a pair to be considered
                            part of the same cluster.
                        Default=0.5
--cls-coverage=<fraction>   Length coverage threshold (0,1].
                        Default=0.7
--cls-one-sided-coverage    Apply coverage threshold to one pair member.
--cls-out-sequences         Output each cluster's sequences in FASTA format.
--cls-algorithm=<code>      0: Complete-linkage clustering;
                            1: Single-linkage clustering.
                        Default=0

Output control options (for search usage except --sort):
-s <TMscore_threshold>      Report results down to this TM-score limit [0,1).
                            0 implies all results are valid for report.
                            NOTE: Also check the pre-screening options below.
                        Default=0.5
--2tm-score                 Include secondary TM-score, 2TM-score: TM-score
                            calculated over matched secondary structures.
--sort=<code>               0: Sort results by the greater TM-score of the two;
                            1: Sort by reference length-normalized TM-score;
                            2: Sort by query length-normalized TM-score;
                            3: Sort by the harmonic mean of the two TM-scores;
                            4: Sort by RMSD.
                            When --2tm-score is set:
                            5: Sort by the greater 2TM-score;
                            6: Sort by reference length-normalized 2TM-score;
                            7: Sort by query length-normalized 2TM-score.
                            8: Sort by the harmonic mean of the 2TM-scores;
                        Default=0
--nhits=<count>             Number of highest-scoring structures to list in
                            the results for each query.
                        Default=2000
--nalns=<count>             Number of highest-scoring structure alignments
                            and superpositions to output for each query.
                        Default=2000
--wrap=<width>              Wrap produced alignments to this width [40,).
                        Default=80
--no-deletions              Remove deletion positions (gaps in query) from
                            produced alignments.
--referenced                Produce transformation matrices for reference
                            structures instead of query(-ies).
--outfmt=<code>             Format of results:
                            0: Plain;
                            1: JSON.
                        Default=0

Interpretation options:
--infmt=<code>              Format of input structures:
                            0: PDB or PDBx/mmCIF, detected automatically;
                            1: PDB;
                            2: PDBx/mmCIF.
                        Default=0
--aatom=<atom_name>         4-character atom name (with spaces) to represent a
                            protein amino acid.
                        Default=" CA "
--natom=<atom_name>         4-character atom name (with spaces) to represent a
                            nucleic acid base.
                        Default=" C3'"
--hetatm                    Consider and align both ATOM and HETATM residues.
--mol=<code>                Molecular type for calculating TM-scores:
                            0: Protein;
                            1: Nucleic acid (RNA, DNA);
                            2: Automatically determined by the majority atom.
                        Default=2
--ter=<code>                Structure file parsing stops at:
                            0: end of file;
                            1: ENDMDL (end of first model) or END of file.
                        Default=1
--split=<code>              Use the following interpretation of a structure
                            file:
                            0: the entire structure is one single chain;
                            1: each MODEL is a separate chain (--ter=0);
                            2: each chain is a separate chain (--ter=0|1).
                        Default=2

Similarity pre-screening options:
--pre-similarity=<similarity_threshold>
                            Minimum pairwise sequence similarity score [0,)
                            for conducting structure comparison.
                            0, all pairs are subject to further processing.
                        Default=0.0
--pre-score=<TMscore_threshold>
                            Minimum provisional TM-score [0,1) for structure
                            pairs to proceed to further stages.
                            0, all pairs are subject to further processing.
                        Default=0.4

Per-pair computation options:
--symmetric                 Always produce symmetric alignments for the same
                            query-reference (reference-query) pair.
--refinement=<code>         Superposition refinement detail level [0,3].
                            Difference between 3 and 1 in TM-score is ~0.001.
                        Default=1
--depth=<code>              Superposition search depth:
                            0: deep;  1: high;  2: medium;  3: shallow.
                        Default=2
--gapcost=<penalty_code>    Gap cost used to estimate local similarity for
                            superposition configurations.
                            0: -0.8;  1: -1.2;  2: -3.0.
                        Default=1
--trigger=<percentage>      Threshold for estimated local similarity in
                            percent [0,100] to trigger superposition
                            analysis for a certain configuration
                            (0, unconditional analysis).
                        Default=50
--seedrule=<code>           Seeding strategy for superposition configurations:
                            0: continuous fragments;
                            1: 64-frame local alignment;
                            2: 128-frame local alignment.
                        Default=1
--window=<size>             Initial window size (in residues) used to analyze
                            superposition candidates {256,512}.
                        Default=256
--recalculate=<count>       Number of top-scored superposition candidates to
                            recalculate accurately {2,4,8,16,32}.
                        Default=32
--nbranches=<number>        Number [0,16] of independent top-performing
                            branches identified during superposition search to
                            explore in more detail (up to --recalculate).
                            NOTE: 0 skips this exploration & refinement.
                        Default=5
--ndps=<count>              Initial number of dynamic programming rounds [0,2]
                            to optimize alignments obtained from superposition
                            search. NOTE: 0 skips and sets --nbranches=0.
                        Default=2
--add-search-by-ss          Include superposition search by a combination of
                            secondary structure and sequence similarity, which
                            helps optimization for some pairs.
--add-chain-level-search    Include chain-level processing for superposition
                            search.
                            NOTE: Active when --chains is set.
--convergence=<number>      Number of final convergence tests [1,30].
                        Default=18

Speed option:
--speed=<code>              Speed up the GTcomplex alignment algorithm at the
                            expense of optimality (larger values => faster;
                            NOTE: the pre-screening options are not affected;
                            NOTE: settings override specified options):
     0: --depth=0 --trigger=0 --nbranches=16 --add-search-by-ss
     1: --depth=0 --trigger=0
     2: --depth=0 --trigger=20
     3: --depth=0 --trigger=50
     4: --depth=1 --trigger=0
     5: --depth=1 --trigger=20
     6: --depth=1 --trigger=50
     7: --depth=2 --trigger=0
     8: --depth=2 --trigger=20
     9: --depth=2 --trigger=50
    10: --depth=2 --trigger=50 --recalculate=16 --nbranches=1 --ndps=1
    11: --depth=3 --trigger=20 --recalculate=16 --nbranches=1 --ndps=1 --refinement=0 --convergence=2
    12: --depth=3 --trigger=50 --recalculate=16 --nbranches=1 --ndps=1 --refinement=0 --convergence=2 --gapcost=2
    13: --depth=3 --trigger=50 --recalculate=8  --nbranches=0 --ndps=1 --refinement=0 --convergence=2 --gapcost=2
    14: --depth=3 --trigger=50 --recalculate=4  --nbranches=0 --ndps=1 --refinement=0 --convergence=2 --gapcost=2
    15: --depth=3 --trigger=50 --recalculate=4  --nbranches=0 --ndps=0 --refinement=0 --convergence=2 --gapcost=2
    16: --depth=3 --trigger=50 --recalculate=2  --nbranches=0 --ndps=0 --refinement=0 --convergence=2 --gapcost=2
                        Default=9

HPC options:
--cpu-threads-reading=<count>
                            Number of CPU threads [1,64] for reading
                            reference data. NOTE that computation on GPU can
                            be faster than reading data by 1 CPU thread.
                        Default=10
--cpu-threads=<count>       Number of CPU threads [1,1024] for parallel
                            computation when compiled without support for
                            GPUs.
                            NOTE: Default number is shown using --dev-list.
                        Default=[MAX(1, #cpu_cores - <cpu-threads-reading>)]
--dev-queries-max-chains=<count>
                            Maximum number of chains [100,256] for query
                            complexes. Higher values increase memory usage and
                            may limit parallel execution.
                        Default=100
--dev-queries-max-per-chunk=<count>
                            Maximum number [1,2] of query complexes (chains
                            when --chains set) processed as a single chunk.
                        Default=1
--dev-queries-total-length-per-chunk=<length>
                            Maximum total length [100,50000] of
                            queries processed as one chunk in parallel.
                            Queries of length larger than the specified
                            length will be skipped. Use large values if
                            required and memory limits permit since they
                            greatly reduce #structure pairs processed in
                            parallel.
                        Default=4000
--dev-max-length=<length>   Maximum length [100,65535] for reference
                            structures. References of length larger than this
                            specified value will be skipped.
                            NOTE: Large values greatly reduce #structure pairs
                            processed in parallel.
                        Default=4000
--dev-min-length=<length>   Minimum length [3,32767] for reference structures.
                            References shorter than this specified value will
                            be skipped.
                        Default=20
--no-file-sort              Do not sort files by size. Data locality can be
                            beneficial when reading files lasts longer than
                            computation.

Device options:
--dev-N=(<number>|,<id_list>)
                            Maximum number of GPUs to use. This can be
                            specified by a number or given by a comma-separated
                            list of GPU identifiers, which should start with a
                            comma. In the latter case, work is distributed in
                            the specified order. Otherwise, more powerful GPUs
                            are selected first.
                            NOTE: The first symbol preceding a list is a comma.
                            NOTE: The option has no effect for the version
                            compiled without support for GPUs.
                        Default=1 (most powerful GPU)
--dev-mem=<megabytes>       Maximum amount of GPU memory (MB) that can be used.
                            All memory is used if a GPU has less than the
                            specified amount of memory.
                        Default=[all memory of GPU(s)] (with support for GPUs)
                        Default=16384 (without support for GPUs)
--dev-expected-length=<length>
                            Expected length of database proteins. Its values
                            are restricted to the interval [20,200].
                            NOTE: Increasing it reduces memory requirements,
                            but mispredictions may cost additional computation
                            time.
                        Default=50
--io-nbuffers=<count>       Number of buffers [2,6] used to cache data read
                            from file. Values greater than 1 lead to increased
                            performance at the expense of increased memory
                            consumption.
                        Default=3
--io-unpinned               Do not use pinned (page-locked) CPU memory.
                            Pinned CPU memory provides better performance, but
                            reduces system resources. If RAM memory is scarce
                            (<2GB), using pinned memory may reduce overall
                            system performance.
                            By default, pinned memory is used.

Other options:
--dev-list                  List all GPUs compatible and available on the
                            system, print a default number for option
                            --cpu-threads (for the CPU version), and exit.
-v [<level_number>]         Verbose mode.
-h                          This text.


Examples:
gtcomplex -v --qrs=str1.cif.gz --rfs=my_huge_structure_database.tar -o my_output_directory
gtcomplex -v --qrs=struct1.pdb --rfs=struct2.pdb,struct3.pdb,struct4.pdb -o my_output_directory
gtcomplex -v --qrs=struct1.pdb,my_struct_directory --rfs=my_ref_directory -o my_output_directory
gtcomplex -v --qrs=str1.pdb.gz,str2.cif.gz --rfs=archive.tar,my_ref_dir --chains -s 0 -o mydir
gtcomplex -v --cls=my_huge_structure_database.tar -o my_output_directory


```
