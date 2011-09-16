//### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
//#
//#   See COPYING file distributed along with the MGTAXA package for the
//#   copyright and license terms.
//#
//### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

////////////////////////////////////////////////////////////////////////////////
//
//  mrsom2: Parallelizing Batch SOM on MR-MPI with supporting for sparse matrix
//
//  Author: Seung-Jin Sul (ssul@jcvi.org)
//
//  Last updated: 09.14.2011
//
////////////////////////////////////////////////////////////////////////////////

#include "mrsom.hpp"

/* -------------------------------------------------------------------------- */
int main(int argc, char** argv)
/* -------------------------------------------------------------------------- */
{
    ///
    /// Read conf file, mrblast.ini and set parameters
    ///
    ifstream config("mrsom.ini");
    if (!config) {
        cerr << "ERROR: configuration file, mrsom.ini, not found" << endl;
        return 1;
    }

    /// parameters
    set<string> options;
    map<string, string> parameters;
    options.insert("*");

    try {
        for (pod::config_file_iterator i(config, options), e ; i != e; ++i) {
            parameters[i->string_key] = i->value[0];
        }

        try {
            SOM_X = boost::lexical_cast<size_t>(parameters["SOMX"]);
            SOM_Y = boost::lexical_cast<size_t>(parameters["SOMY"]);
            SOM_D = boost::lexical_cast<size_t>(parameters["SOMD"]);
        }
        catch (const boost::bad_lexical_cast &) {
            cerr << "Exception: bad_lexical_cast" << endl;
        }
    }
    catch (exception& e) {

        cerr << "Exception: " << e.what() << endl;
    }

    po::options_description generalDesc("General options");
    generalDesc.add_options()
    ("help", "print help message")
    ("mode,m", po::value<string>(), "set train/test mode, \"train or test\"")
    ("page-size,p", po::value<int>(&SZPAGE)->default_value(64),
     "[OPTIONAL] set page size of MR-MPI (default=64MB)")
    ;

    po::options_description trainnigDesc("Options for training");
    trainnigDesc.add_options()
    ("infile,i", po::value<string>(), "set input file name")
    ("outfile,o", po::value<string>(&OUTPREFIX)->default_value("result"),
     "set a prefix for outout file name")
    ("nepochs,e", po::value<unsigned int>(), "set the number of iterations")
    ("nvecs,n", po::value<uint32_t>(), "set the number of feature vectors")
    ("ndim,d", po::value<uint32_t>(), "set the number of dimension of input feature vector")
    ("nblocks,b", po::value<uint32_t>(), "set the number of blocks")
    ("sparse,s", po::value<int>(&bSPARSE)->default_value(0),
     "[OPTIONAL] sparse matrix as input or not (default=0)")    
    ;
    
    string binFileName, indexFileName, numFileName;
    string somMapFileName;
    ifstream inputBinFile, inputIndexFile;
    
    po::options_description trainnigSparseDesc("Options for training (sparse matrix)");
    trainnigSparseDesc.add_options()
    ("indexfile,x", po::value<string>(), 
     "set input index file name")
    ("numfile,t", po::value<string>(), 
     "set input total num values file name")
    ;
    
    po::options_description testingDesc("Options for testing");
    testingDesc.add_options()
    ("codebook,c", po::value<string>(), "set saved codebook file name")
    ;

    po::options_description allDesc("Allowed options");
    allDesc.add(generalDesc).add(trainnigDesc).add(trainnigSparseDesc).add(testingDesc);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, allDesc), vm);
    po::notify(vm);

    
    string ex = "Example for normal matrix\n";
    ex += "  Converting ASCII input file to bin: txt2bin rgbs.txt rgbs.bin 3 28\n";
    ex += "  Training: mpirun -np 4 mrsom -m train -i rgbs.bin -o rgbs -e 10 -n 28 -d 3 -b 4\n";
    ex += "  Testing:  mrsom -m test -c rgbs-codebook.txt -i rgbs.txt -o rgbs -d 3 -n 10 \n\n";
    
    string ex2= "Example for sparse matrix\n";
    ex2 += "  Training: mpirun -np 4 mrsom -s 1 -m train -i rgbs-sparse.bin -x rgbs-sparse.idx -t rgbs-sparse.num -o rgbs-sparse -e 10 -n 28 -d 3 -b 4\n";
    ex2 += "  Testing:  mrsom -s 1 -m test -c rgbs-sparse-codebook.txt -i rgbs-sparse.txt -o rgbs-sparse -d 3 -n 10 \n\n";

    if (argc < 2 || (!strcmp(argv[1], "-?") || !strcmp(argv[1], "--?")
                 || !strcmp(argv[1], "/?") || !strcmp(argv[1], "/h")
                 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--h")
                 || !strcmp(argv[1], "--help") || !strcmp(argv[1], "/help")
                 || !strcmp(argv[1], "-help")  || !strcmp(argv[1], "help"))) {
        cout << "MR-MPI Batch SOM\n"
             << "\nAuthor: Seung-Jin Sul (ssul@jcvi.org)\n"
             << allDesc << "\n" << ex << ex2;
        return 1;
    }
    else {
        /// OPTIONAL
        if (vm.count("page-size")) SZPAGE = vm["page-size"].as<int>();
        if (vm.count("outfile")) OUTPREFIX = vm["outfile"].as<string>();
        if (vm.count("sparse")) bSPARSE = vm["sparse"].as<int>();       
        
        /// MANDATORY
        if (vm.count("infile") && vm.count("nvecs") && vm.count("ndim") && vm.count("mode")) {
            binFileName = vm["infile"].as<string>();
            NVECS = vm["nvecs"].as<unsigned int>();
            NDIMEN = vm["ndim"].as<unsigned int>();
            string trainOrTest = vm["mode"].as<string>();
            
            if (!trainOrTest.compare("train")) {
                RUNMODE = TRAIN;
                if (vm.count("nepochs")) NEPOCHS = vm["nepochs"].as<unsigned int>();
                if (vm.count("nblocks")) {
                    NBLOCKS = vm["nblocks"].as<unsigned int>();
                    /// Note: The number of allocated vectors for the last work item 
                    /// will be adjusted to NVECSPERRANK + NVECSLEFT if NVECSLEFT != 0
                }
                if (bSPARSE) {
                    if (vm.count("indexfile") && vm.count("numfile")) {
                        indexFileName = vm["indexfile"].as<string>();
                        numFileName = vm["numfile"].as<string>();
                    }
                    else {
                        cout << "Option error: sparse mode error" << "\n" << ex << ex2;
                        return 1;
                    }
                }
            }
            else if (!trainOrTest.compare("test")) {
                RUNMODE = TEST;
                if (vm.count("codebook")) somMapFileName = vm["codebook"].as<string>();
                else {
                    cout << "Option error: testing needs codebook" << "\n" << ex << ex2;
                    return 1;
                }                
            }    
        }
        else {
            cout << allDesc << "\n" << ex << ex2;
            return 1;
        }
    }
 

    ///
    /// Read input vector file
    ///
    read_matrix(binFileName.c_str(), indexFileName.c_str());
    
    ///
    /// Codebook resize
    ///
    CODEBOOK.resize(boost::extents[SOM_Y][SOM_X][NDIMEN]);
    NUMER1.resize(boost::extents[SOM_Y * SOM_X * NDIMEN]);
    DENOM1.resize(boost::extents[SOM_Y * SOM_X]);
    NUMER2.resize(boost::extents[SOM_Y * SOM_X * NDIMEN]);
    DENOM2.resize(boost::extents[SOM_Y * SOM_X]);
    
    ///
    /// TESTING MODE
    ///
    if (RUNMODE == TEST) {
        test(somMapFileName.c_str(), binFileName.c_str());
        return 0;
    }

    ///
    /// Fill initial random weights
    ///
    init_codebook((unsigned int)time(0));
       
    ///
    /// MPI init
    ///
    int MPI_myId, MPI_nProcs, MPI_length;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_myId);
    g_RANKID = MPI_myId;
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_nProcs);
    MPI_Barrier(MPI_COMM_WORLD);
    double profile_time = MPI_Wtime();
    
    ///
    /// MR-MPI
    ///
    MapReduce* mr = new MapReduce(MPI_COMM_WORLD);
    /*
    * mapstyle = 0 (chunk) or 1 (stride) or 2 (master/slave)
    * all2all = 0 (irregular communication) or 1 (use MPI_Alltoallv)
    * verbosity = 0 (none) or 1 (summary) or 2 (histogrammed)
    * timer = 0 (none) or 1 (summary) or 2 (histogrammed)
    * memsize = N = number of Mbytes per page of memory
    * minpage = N = # of pages to pre-allocate per processor
    * maxpage = N = max # of pages allocatable per processor, 0 = no limit
    * keyalign = N = byte-alignment of keys
    * valuealign = N = byte-alignment of values
    * fpath = string
    */
    mr->verbosity = 0;
    mr->timer = 0;
    mr->mapstyle = 0;               /// NOTE: MPI_reduce() does not work with master/slave mode
    mr->memsize = SZPAGE;           /// page size
    mr->keyalign = sizeof(uint32_t);/// default: key type = uint32_t = 8 bytes
    MPI_Barrier(MPI_COMM_WORLD);

    ///
    /// Parameters for SOM
    ///
    FLOAT_T N = (FLOAT_T)NEPOCHS;   /// iterations
    FLOAT_T nrule, nrule0 = 0.9f;   /// learning rate factor
    FLOAT_T R0;
    R0 = SOM_X / 2.0f;              /// init radius for updating neighbors
    R = R0;
    unsigned int x = 0;             /// 0...N-1
    
    ///
    /// Set NVECSPERRANK according to NBLOCKS
    ///
    /// Note: The number of allocated vectors for the last work item 
    /// will be adjusted to NVECSPERRANK + NVECSLEFT if NVECSLEFT != 0
    ///
    if (NBLOCKS < MPI_nProcs || NBLOCKS % MPI_nProcs) {
        cerr << "ERROR: please select nblocks as nblocks >= ncores and nblocks % ncores == 0.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    /// 
    /// reinterpret_cast memmapped bin file and set NVECSPERRANK and NVECSLEFT
    ///
    if (bSPARSE) {
        assert(MMAPBINFILE.is_open());
        assert(MMAPIDXFILE.is_open());
        
        FDATASPARSE = reinterpret_cast<SPARSE_STRUCT_T*>((char*)MMAPBINFILE.data());   
        INDEXSPARSE = reinterpret_cast<INDEX_STRUCT_T*>((char*)MMAPIDXFILE.data());   
        ///
        /// For sparse matrix
        /// 1. read *.num for getting the total number of values in the matrix
        ///    (*.num file is generated from txt2bin-sparse tool)               
        /// 2. compute NVALSSPERRANK by NTOTALVALUES / NBLOCKS
        /// 3. set work item start and end using NVALSSPERRANK
        ///
        ifstream numFile(numFileName.c_str());
        uint32_t numValues;
        numFile >> numValues;
        numFile.close();
        
        uint32_t numValuesPerBlock = ceil(numValues / NBLOCKS);
        uint32_t numLefty = numValues % numValuesPerBlock;
        uint32_t rowStart = 0;
        uint32_t rowEnd = 0;
        
        for (uint32_t i = 0; i < NVECS; i++) {
            uint32_t numConsidered = 0;
            while (numConsidered <= numValuesPerBlock) {
                numConsidered += (INDEXSPARSE+i)->num_values;
                i++;
                if (i >= NVECS) break;            
            }    
            rowEnd = i-1;
            SPARSEWORKITEM_STRUCT_T bBlock;
            bBlock.start = rowStart;
            bBlock.end = rowEnd;
            g_vecSparseWorkItem.push_back(bBlock);
            rowStart = i;        
        }        
    }
    else {
        assert(MMAPBINFILE.is_open());
        FDATA = reinterpret_cast<FLOAT_T*>((char*)MMAPBINFILE.data());   
        NVECSPERRANK = ceil(NVECS / NBLOCKS);
        NVECSLEFT = NVECS % NBLOCKS; /// The last work item will be assigned NVECSPERRANK + NVECSLEFT vectors
    }
    
    ///
    /// Training
    ///
    while (NEPOCHS && R > 1.0) {
        if (MPI_myId == 0) {
            R = R0 * exp(-10.0f * (x * x) / (N * N));
            x++;
            printf("Epoch: %d   R: %.2f \n", (NEPOCHS - 1), R);
        }
        MPI_Bcast(&R, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (SZFLOAT == 4)
            MPI_Bcast((void*)CODEBOOK.data(), SOM_Y * SOM_X * NDIMEN, MPI_FLOAT, 0, MPI_COMM_WORLD);
        else if (SZFLOAT == 8)
            MPI_Bcast((void*)CODEBOOK.data(), SOM_Y * SOM_X * NDIMEN, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /// v9 using MPI_reduce
        ///
        /// 1. Each task fills NUMER1 and DENOM1
        /// 2. MPI_reduce sums up each tasks NUMER1 and DENOM1 to the root's
        ///    NUMER2 and DENOM2.
        /// 3. Update CODEBOOK using NUMER2 and DENOM2
        ///
        if (MPI_myId == 0) {
            for (size_t y = 0; y < SOM_Y; y++) {
                for (size_t x = 0; x < SOM_X; x++) {
                    DENOM2[y * SOM_X + x] = 0.0;
                    for (size_t d = 0; d < NDIMEN; d++) {
                        NUMER2[y * SOM_X * NDIMEN + x * NDIMEN + d] = 0.0;
                    }
                }
            }
        }

        if (bSPARSE) mr->map(NBLOCKS, &mpireduce_train_batch_sparse, NULL);
        else         mr->map(NBLOCKS, &mpireduce_train_batch, NULL);
        
        if (MPI_myId == 0) {
            for (size_t y = 0; y < SOM_Y; y++) {
                for (size_t x = 0; x < SOM_X; x++) {
                    FLOAT_T denom = DENOM2[y * SOM_X + x];
                    for (size_t d = 0; d < NDIMEN; d++) {
                        FLOAT_T newWeight = NUMER2[y * SOM_X * NDIMEN + x * NDIMEN + d] / denom;
                        if (newWeight > 0.0) CODEBOOK[y][x][d] = newWeight;
                    }
                }
            }
        }
        NEPOCHS--;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    ///
    /// Save u-matrix a codebook
    ///
    if (MPI_myId == 0) {
        printf("INFO: Saving SOM map and U-Matrix...\n");
        
        string umatFileName = OUTPREFIX + "-umat.txt";                
        cout << "\tSaving U-mat file = " << umatFileName << endl;
        int ret = save_umat(umatFileName.c_str());
        if (ret < 0) printf("    Failed to save u-matrix. !\n");
        
        string cbFileName = OUTPREFIX + "-codebook.txt";
        cout << "\tCodebook file = " << cbFileName << endl;
        save_codebook(cbFileName.c_str());
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (bSPARSE) {
        inputIndexFile.close();
        MMAPIDXFILE.close();
    }
    MMAPBINFILE.close();
    delete mr;

    profile_time = MPI_Wtime() - profile_time;
    if (MPI_myId == 0) {
        cerr << "Total Execution Time: " << profile_time << endl;
    }

    MPI_Finalize();
    
    return 0;
}



/** MR-MPI user-defined map function - batch training with MPI_reduce()
 * @param itask - number of work items
 * @param kv
 * @param ptr
 */

void mpireduce_train_batch(int itask,
                           KeyValue* kv,
                           void* ptr)
{
    int p1[SOM_D];
    int p2[SOM_D];

    /// Initialize DENOM1 and NUMER1
    for (size_t y = 0; y < SOM_Y; y++) {
        for (size_t x = 0; x < SOM_X; x++) {
            DENOM1[y * SOM_X + x] = 0.0;
            for (size_t d = 0; d < NDIMEN; d++) {
                NUMER1[y * SOM_X * NDIMEN + x * NDIMEN + d] = 0.0;
            }
        }
    }
    
    uint32_t nvecs = NVECSPERRANK;
    /// Do NVECSPERRANK + NVECSLEFT if NVECSLEFT != 0 for the last work item
    if (itask == NBLOCKS - 1 && NVECSLEFT != 0) nvecs = NVECSPERRANK + NVECSLEFT;
    
    for (uint32_t n = 0; n < nvecs; n++) {
        /// get the coords of the best matching unit 
        get_bmu_coord(p1, itask, n);

        /// Accumulate denoms and numers
        for (size_t y = 0; y < SOM_Y; y++) {
            for (size_t x = 0; x < SOM_X; x++) {
                p2[0] = x;
                p2[1] = y;
                FLOAT_T dist = 0.0f;
                for (size_t p = 0; p < SOM_D; p++) {
                    dist += (p1[p] - p2[p]) * (p1[p] - p2[p]);
                }
                dist = sqrt(dist);

                FLOAT_T neighbor_fuct = 0.0f;
                neighbor_fuct = exp(-(1.0f * dist * dist) / (R * R));

                for (size_t d = 0; d < NDIMEN; d++) {
                    FLOAT_T v = *((FDATA + itask * NDIMEN * NVECSPERRANK) + n * NDIMEN + d);
                    NUMER1[y * SOM_X * NDIMEN + x * NDIMEN + d] += 1.0f * neighbor_fuct * v;
                }
                DENOM1[y * SOM_X + x] += neighbor_fuct;
            }
        }
    }

    if (SZFLOAT == 4) { /// 4 bytes float
        MPI_Reduce((void*)NUMER1.data(), (void*)NUMER2.data(),
                   SOM_Y * SOM_X * NDIMEN, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce((void*)DENOM1.data(), (void*)DENOM2.data(),
                   SOM_Y * SOM_X, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    else if (SZFLOAT == 8) { /// 8 bytes double
        MPI_Reduce((void*)NUMER1.data(), (void*)NUMER2.data(),
                   SOM_Y * SOM_X * NDIMEN, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce((void*)DENOM1.data(), (void*)DENOM2.data(),
                   SOM_Y * SOM_X, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
}


/** MR-MPI user-defined map function - batch training with MPI_reduce()
 * @param itask - number of work items
 * @param kv
 * @param ptr
 */

void mpireduce_train_batch_sparse(int itask,
                                  KeyValue* kv,
                                  void* ptr)
{  
    int p1[SOM_D];
    int p2[SOM_D];

    /// Initialize DENOM1 and NUMER1
    for (size_t y = 0; y < SOM_Y; y++) {
        for (size_t x = 0; x < SOM_X; x++) {
            DENOM1[y * SOM_X + x] = 0.0;
            for (size_t d = 0; d < NDIMEN; d++) {
                NUMER1[y * SOM_X * NDIMEN + x * NDIMEN + d] = 0.0;
            }
        }
    }
    
    /// row start~end for the work item assigned
    uint32_t rowStart = g_vecSparseWorkItem[itask].start;
    uint32_t rowEnd = g_vecSparseWorkItem[itask].end;
    
    for (uint32_t n = rowStart; n < rowEnd+1; n++) {
        /// get the best matching unit
        get_bmu_coord(p1, itask, n);
        
        /// Accumulate denoms and numers
        for (size_t y = 0; y < SOM_Y; y++) {
            for (size_t x = 0; x < SOM_X; x++) {
                p2[0] = x;
                p2[1] = y;
                FLOAT_T dist = 0.0f;
                for (size_t p = 0; p < SOM_D; p++) {
                    dist += (p1[p] - p2[p]) * (p1[p] - p2[p]);
                }
                dist = sqrt(dist);

                FLOAT_T neighbor_fuct = 0.0f;
                neighbor_fuct = exp(-(1.0f * dist * dist) / (R * R));

                uint32_t numValues = (INDEXSPARSE + n)->num_values;
                uint32_t numValuesAcc = (INDEXSPARSE + n)->num_values_accum;
                uint32_t dataLoc = numValuesAcc - numValues; 
                size_t d2 = 0;
                for (size_t d = 0; d < NDIMEN; d++) {
                    FLOAT_T v = 0.0;
                    if ((FDATASPARSE + dataLoc + d2)->index == d)  {
                        v = (FDATASPARSE + dataLoc + d2)->value;
                        d2++;
                        NUMER1[y * SOM_X * NDIMEN + x * NDIMEN + d] += 1.0f * neighbor_fuct * v;
                    }
                }
                DENOM1[y * SOM_X + x] += neighbor_fuct;
            }
        }
    }
    
    if (SZFLOAT == 4) { /// 4 bytes float
        MPI_Reduce((void*)NUMER1.data(), (void*)NUMER2.data(), SOM_Y * SOM_X * NDIMEN, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce((void*)DENOM1.data(), (void*)DENOM2.data(), SOM_Y * SOM_X, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    else if (SZFLOAT == 8) { /// 8 bytes double
        MPI_Reduce((void*)NUMER1.data(), (void*)NUMER2.data(), SOM_Y * SOM_X * NDIMEN, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce((void*)DENOM1.data(), (void*)DENOM2.data(), SOM_Y * SOM_X, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
}



/** MR-MPI Map function - Get node coords for the best matching unit (BMU)
 * @param coords - BMU coords
 * @param itask - #task
 * @param n - row num in the input feature file
 */

void get_bmu_coord(int* coords,
                   int itask,
                   uint32_t rownum)
{
    FLOAT_T mindist = std::numeric_limits<FLOAT_T>::max();
    FLOAT_T dist = 0.0f;

    ///
    /// Check SOM_X * SOM_Y nodes one by one and compute the distance
    /// D(W_K, Fvec) and get the mindist and get the coords for the BMU.
    ///
    for (size_t y = 0; y < SOM_Y; y++) {
        for (size_t x = 0; x < SOM_X; x++) {
            /// dist = get_distance(CODEBOOK[y][x],
            /// (FDATA + itask*NDIMEN*NVECSPERRANK) + n*NDIMEN, DISTOPT);
            dist = get_distance(y, x, itask, rownum, DISTOPT);
            if (dist < mindist) {
                mindist = dist;
                coords[0] = x;
                coords[1] = y;
            }
        }
    }
}


/** MR-MPI Map function - Distance b/w a feature vector and a weight vector
 * = Euclidean
 * @param y
 * @param x
 * @param r - row number in the input feature file
 * @param distance_metric
 */

FLOAT_T get_distance(size_t y,
                     size_t x,
                     int itask,
                     size_t rownum,
                     unsigned int distance_metric)
{
    FLOAT_T distance = 0.0f;
    if (bSPARSE) {
        uint32_t numValues = (INDEXSPARSE+rownum)->num_values;
        uint32_t numValuesAcc = (INDEXSPARSE+rownum)->num_values_accum;
        uint32_t dataLoc = numValuesAcc - numValues;
        size_t d2 = 0;
        for (size_t d = 0; d < NDIMEN; d++) {
            /// *((FDATASPARSE + itask * NDIMEN * NVECSPERRANK) + rownum * NDIMEN + d)
            FLOAT_T v = 0.0;
            if ((FDATASPARSE+dataLoc+d2)->index == d)  {
                v = (FDATASPARSE+dataLoc+d2)->value;
                d2++;
            }
            distance += (CODEBOOK[y][x][d] - v) * (CODEBOOK[y][x][d] - v);
        }
        return sqrt(distance);
    }
    else {
        switch (distance_metric) {
        default:
        case 0: /// EUCLIDIAN
            for (size_t d = 0; d < NDIMEN; d++) {
                FLOAT_T v = *((FDATA + itask * NDIMEN * NVECSPERRANK) + rownum * NDIMEN + d);
                distance += (CODEBOOK[y][x][d] - v) * (CODEBOOK[y][x][d] - v);
            }
            return sqrt(distance);
        //case 1: /// SOSD: //SUM OF SQUARED DISTANCES
            //if (m_weights_number >= 4) {
                //distance = mse(vec, m_weights, m_weights_number);
            //} else {
            //for (unsigned int d = 0; d < NDIMEN; d++)
                //distance += (vec[d] - wvec[d]) * (vec[d] - wvec[d]);
            //}
            //return distance;
        //case 2: /// TXCB: //TAXICAB
            //for (unsigned int d = 0; d < NDIMEN; d++)
                //distance += fabs(vec[d] - wvec[d]);
            //return distance;
        //case 3: /// ANGL: //ANGLE BETWEEN VECTORS
            //for (unsigned int d = 0; d < NDIMEN; d++) {
                //distance += vec[d] * wvec[d];
                //n1 += vec[d] * vec[d];
                //n2 += wvec[d] * wvec[d];
            //}
            //return acos(distance / (sqrt(n1) * sqrt(n2)));
        //case 4: /// MHLN:   //mahalanobis
            //distance = sqrt(m_weights * cov * vec)
            //return distance
        }
    }
}


/** MR-MPI Map function - Distance b/w vec1 and vec2, default distance_metric
 * = Euclidean
 * @param vec1
 * @param vec2
 * @param distance_metric:
 */

FLOAT_T get_distance(const FLOAT_T* vec1,
                     const FLOAT_T* vec2,
                     unsigned int distance_metric)
{
    FLOAT_T distance = 0.0f;
    FLOAT_T n1 = 0.0f, n2 = 0.0f;
    switch (distance_metric) {
    default:
    case 0: /// EUCLIDIAN
        for (size_t d = 0; d < NDIMEN; d++) {
            distance += (vec1[d] - vec2[d]) * (vec1[d] - vec2[d]);
        }
        return sqrt(distance);
    //case 1: /// SOSD: //SUM OF SQUARED DISTANCES
        //if (m_weights_number >= 4) {
            //distance = mse(vec, m_weights, m_weights_number);
        //} else {
        //for (unsigned int d = 0; d < NDIMEN; d++)
            //distance += (vec[d] - wvec[d]) * (vec[d] - wvec[d]);
        //}
        //return distance;
    //case 2: /// TXCB: //TAXICAB
        //for (unsigned int d = 0; d < NDIMEN; d++)
            //distance += fabs(vec[d] - wvec[d]);
        //return distance;
    //case 3: /// ANGL: //ANGLE BETWEEN VECTORS
        //for (unsigned int d = 0; d < NDIMEN; d++) {
            //distance += vec[d] * wvec[d];
            //n1 += vec[d] * vec[d];
            //n2 += wvec[d] * wvec[d];
        //}
        //return acos(distance / (sqrt(n1) * sqrt(n2)));
    //case 4: /// MHLN:   //mahalanobis
        //distance = sqrt(m_weights * cov * vec)
        //return distance
    }
}

/** MR-MPI Map function - Distance b/w vec and weight vector in the codebook
 * ,default distance_metric = Euclidean
 * @param somy
 * @param somx
 * @param vec2
 * @param distance_metric:
 */

FLOAT_T get_distance(size_t somy,
                     size_t somx,
                     const FLOAT_T* vec,
                     unsigned int distance_metric)
{
    FLOAT_T distance = 0.0f;
    FLOAT_T n1 = 0.0f, n2 = 0.0f;
    switch (distance_metric) {
    default:
    case 0: /// EUCLIDIAN
        for (size_t d = 0; d < NDIMEN; d++) {
            distance += (vec[d] - CODEBOOK[somy][somx][d]) * (vec[d] - CODEBOOK[somy][somx][d]);
        }
        return sqrt(distance);
    }
}


/** MR-MPI Map function - Get weight vector from CODEBOOK using x, y index
 * @param y - y coordinate of a node in the map
 * @param x - x coordinate of a node in the map
 */

FLOAT_T* get_wvec(size_t y,
                  size_t x)
{
    FLOAT_T* wvec = (FLOAT_T*)malloc(SZFLOAT * NDIMEN);
    for (size_t d = 0; d < NDIMEN; d++)
        wvec[d] = CODEBOOK[y][x][d]; /// NOTE: (y,x) order
    return wvec;
}

/** Save u-matrix
 * @param fname
 */

int save_umat(const char* fname)
{
    int D = 2;
    FLOAT_T min_dist = 1.5f;
    FILE* fp = fopen(fname, "wt");
    if (fp != 0) {
        unsigned int n = 0;
        for (size_t som_y1 = 0; som_y1 < SOM_Y; som_y1++) {
            for (size_t som_x1 = 0; som_x1 < SOM_X; som_x1++) {
                FLOAT_T dist = 0.0f;
                unsigned int nodes_number = 0;
                int coords1[2];
                coords1[0] = som_x1;
                coords1[1] = som_y1;

                for (size_t som_y2 = 0; som_y2 < SOM_Y; som_y2++) {
                    for (size_t som_x2 = 0; som_x2 < SOM_X; som_x2++) {
                        unsigned int coords2[2];
                        coords2[0] = som_x2;
                        coords2[1] = som_y2;

                        if (som_x1 == som_x2 && som_y1 == som_y2) 
                            continue;

                        FLOAT_T tmp = 0.0;
                        for (size_t d = 0; d < D; d++) {
                            tmp += pow(coords1[d] - coords2[d], 2.0f);
                        }
                        tmp = sqrt(tmp);
                        if (tmp <= min_dist) {
                            nodes_number++;
                            FLOAT_T* vec1 = get_wvec(som_y1, som_x1);
                            FLOAT_T* vec2 = get_wvec(som_y2, som_x2);
                            dist += get_distance(vec1, vec2, DISTOPT);
                            free(vec1);
                            free(vec2);
                        }
                    }
                }
                dist /= (FLOAT_T)nodes_number;
                if (isnan(dist)) dist = 0.0;
                fprintf(fp, " %f", dist);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        return 0;
    }
    else
        return -2;
}



/** Classify - Compute BMU for new test vectors on the trained SOM MAP. The
 * output is the coords (x, y) in the som map.
 * @param vec
 * @param p
 */

void classify(const FLOAT_T* vec, 
              int* p)
{
    FLOAT_T mindist = std::numeric_limits<FLOAT_T>::max();
    FLOAT_T dist = 0.0f;

    ///
    /// Check SOM_X * SOM_Y nodes one by one and compute the distance
    /// D(W_K, Fvec) and get the mindist and get the coords for the BMU.
    ///
    for (size_t y = 0; y < SOM_Y; y++) {
        for (size_t x = 0; x < SOM_X; x++) {
            dist = get_distance(y, x, vec, DISTOPT);
            if (dist < mindist) {
                mindist = dist;
                p[0] = x;
                p[1] = y;
            }
        }
    }
}


/** Initialize codebook
 * @param seed
 */
 
void init_codebook(unsigned int seed)
{
    srand(seed);
    for (size_t y = 0; y < SOM_Y; y++) {
        for (size_t x = 0; x < SOM_X; x++) {
            for (size_t d = 0; d < NDIMEN; d++) {
                int w = 0xFFF & rand();
                w -= 0x800;
                CODEBOOK[y][x][d] = (FLOAT_T)w / 4096.0f;
            }
        }
    }
}

/** Save codebook to file
 * @param fname
 */
 
int save_codebook(const char* cbFileName)
{
    char temp[80];
    ofstream mapFile2(cbFileName);
    if (mapFile2.is_open()) {
        for (size_t y = 0; y < SOM_Y; y++) {
            for (size_t x = 0; x < SOM_X; x++) {
                for (size_t d = 0; d < NDIMEN; d++) {
                    sprintf(temp, "%0.10f", CODEBOOK[y][x][d]);
                    mapFile2 << temp << "\t";
                }
            }
            mapFile2 << endl;
        }
        mapFile2.close();
        return 0;
    }
    else {
        return 1;
    }
}

/** Load codebook from file
 * @param fname
 */
 
int load_codebook(const char *mapFilename)
{
    FILE* somMapFile = fopen(mapFilename, "r");
    if (somMapFile) {
        for (size_t y = 0; y < SOM_Y; y++) {
            for (size_t x = 0; x < SOM_X; x++) {
                for (size_t d = 0; d < NDIMEN; d++) {
                    FLOAT_T tmp = 0.0f;
                    fscanf(somMapFile, "%f", &tmp);
                    CODEBOOK[y][x][d] = tmp;
                }
            }
        }
    }
    else {
        return 1;
    }
    fclose(somMapFile);
    return 0;
}


/** read_matrix: memmapping input matrix file
 * @param fname
 */
 
void read_matrix(const char *filename, const char *idxfilename)
{
    ///
    /// If sparse, load index file which contains the star position of each
    /// row and the number of values. 
    ///    
    if (bSPARSE) {
        unsigned long int realFileSize = boost::filesystem::file_size(idxfilename);
        MMAPIDXFILE.open(idxfilename, realFileSize, 0);
        if (!MMAPIDXFILE.is_open()) {
            cerr << "ERROR: failed to open mmap index file\n";
            MPI_Finalize();
            exit(1);
        }
    }
    
    unsigned long int realFileSize = boost::filesystem::file_size(filename);
    MMAPBINFILE.open(filename, realFileSize, 0);
    if (!MMAPBINFILE.is_open()) {
        cerr << "ERROR: failed to open mmap bin file\n";
        MPI_Finalize();
        exit(1);
    }
}

/** test
 * @param codebook
 * @param infilename
 */
 
void test(const char* codebook, 
          const char* binFileName) 
{
    ///
    /// Load codebook
    ///
    if (load_codebook(codebook) > 0) {
        cerr << "ERROR: codebook load error.\n";
        exit(0);
    }

    ///
    /// Classification: get the coords of the trained SOM MAP for new
    /// vectors for testing.
    ///
    string classFileName = OUTPREFIX + "-class.txt";
    FILE* classOutFile = fopen(classFileName.c_str(), "w");
    int p[SOM_D];
    
    FILE* testingFile = fopen(binFileName, "r");
    FLOAT_T vec[NDIMEN];        
    if (classOutFile && testingFile) {
        for (size_t nvec = 0; nvec < NVECS; nvec++) {
            for (size_t d = 0; d < NDIMEN; d++) {
                FLOAT_T tmp = 0.0f;
                fscanf(testingFile, "%f", &tmp);
                vec[d] = tmp;
            }
            /////////////////
            classify(vec, p);
            /////////////////
            fprintf(classOutFile, "%d\t%d\n", p[0], p[1]); /// somx,somy
        }
    }
    else {
        cerr << "ERROR: test file open error.\n";
        exit(0);
    }
    fclose(testingFile);
    fclose(classOutFile);     
}
 

/// EOF
