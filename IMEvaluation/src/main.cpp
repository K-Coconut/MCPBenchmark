#define HEAD_INFO

#include "sfmt/SFMT.h"
#include "memoryusage.h"
#include "head.h"
#include "string.h"
using namespace std;
namespace fs = std::experimental::filesystem;

class Argument
{
public:
    string seedFile;
    string outputFile;
    string graphFile;
    int seed_random;
    int64 size;
    int numIters;
    vector<int> klist;
};

#include "graph.h"
#include "infgraph.h"

void writeResult(string file, float coverage)
{
    ofstream outputFile;
    outputFile.open(file.c_str());
    outputFile << coverage;
    outputFile.close();
}

vector<int> readSeedSet(string file)
{
    ifstream infile((file).c_str());
    string line;
    vector<int> seeds;
    while (getline(infile, line))
    {
        int a;
        if (line.empty())
        {
            break;
        }
        a = stoi(line.substr(0, line.find("\n")));
        seeds.push_back(a);
    }
    return seeds;
}

void run_with_parameter(InfGraph &g, const Argument &arg)
{
    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "To generate " << arg.size << " RR sets" << endl;
    clock_t time_start = clock();
    g.build_hyper_graph_r(arg);
    clock_t time_end = clock();
    cout << "RR sets generated, costs: " << (float)(time_end - time_start) / CLOCKS_PER_SEC << " sec" << endl;

    if (arg.klist.size() != 0) {
        for (auto k : arg.klist) {
            string inputFile, outputFile;
            char buff[1000];
            snprintf(buff, sizeof(buff), arg.seedFile.c_str(), k);
            inputFile = buff;
            char buff2[1000];
            snprintf(buff2, sizeof(buff2), arg.outputFile.c_str(), k);
            outputFile = buff2;
            vector<int> seeds = readSeedSet(inputFile);
            float coverage = g.InfluenceHyperGraph(seeds, arg);
            cout << " budget: " << k << " coverage: " << coverage << endl;
            writeResult(outputFile, coverage);
        }

    } else {
        string inputFile, outputFile;
        inputFile = arg.seedFile;
        outputFile = arg.outputFile;
        vector<int> seeds = readSeedSet(inputFile);
        float coverage = g.InfluenceHyperGraph(seeds, arg);
        cout << "coverage: " << coverage << endl;
        writeResult(outputFile, coverage);
        Timer::show();
    }
}
void Run(int argn, char **argv)
{
    Argument arg;
    for (int i = 0; i < argn; i++)
    {
        if (argv[i] == string("-help") || argv[i] == string("--help") || argn == 1)
        {
            cout << "./imm -dataset *** -size size of RR sets -num number of iterations" << endl;
            return;
        }
        if (argv[i] == string("-seedFile"))
            arg.seedFile = argv[i + 1];

        if (argv[i] == string("-output"))
            arg.outputFile = argv[i + 1];

        if (argv[i] == string("-graphFile"))
            arg.graphFile = argv[i + 1];

        if (argv[i] == string("-klist")) {
            vector<int> klist;
            string s = argv[i + 1];
            string delimiter = ",";
            size_t pos = 0;
            int budget;
            string all = "[";
            while ((pos = s.find(delimiter)) != std::string::npos) {
                string k = s.substr(0, pos);
                budget = atoi(k.c_str());
                klist.push_back(budget);
                all += k;
                all += ",";
                s.erase(0, pos + delimiter.length());
            }
            all += "]";
            cout << "budgets: " << all << endl;
            arg.klist = klist;
        }
        
        if (argv[i] == string("-size"))
            arg.size = atoi(argv[i + 1]);
    }
    ASSERT(arg.seedFile != "" && arg.outputFile != "");

    string graph_file = arg.graphFile;
    std::size_t found = arg.graphFile.find_last_of("/");
    string folder = arg.graphFile.substr(0, found + 1);
    InfGraph g(folder, graph_file);
    g.setInfuModel(InfGraph::IC);

    run_with_parameter(g, arg);
}

int main(int argn, char **argv)
{
    __head_version = "v1";
    OutputInfo info(argn, argv);

    Run(argn, argv);
}
