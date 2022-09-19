#define HEAD_INFO
//#define HEAD_TRACE
#include "head.h"
typedef double (*pf)(int, int);
class Graph
{
public:
    int n, m, k;
    vector<int> inDeg;
    vector<int> outDeg;
    vector<vector<int>> neighbors;

    vector<vector<double>> probT;

    enum InfluModel {IC, LT, CONT};
    InfluModel influModel;
    void setInfuModel(InfluModel p)
    {
        influModel = p;
        TRACE(influModel == IC);
        TRACE(influModel == LT);
        TRACE(influModel == CONT);
    }

    string folder;
    string graph_file;
    int num_k;
    void readNM()
    {
        cout << ":" << folder << ":" << endl;
        ifstream cin((folder + "attribute.txt").c_str());
        ASSERT(!cin == false);
        string s;
        while (cin >> s)
        {
            if (s.substr(0, 2) == "n=")
            {
                n = atoi(s.substr(2).c_str());
                continue;
            }
            if (s.substr(0, 2) == "m=")
            {
                m = atoi(s.substr(2).c_str());
                continue;
            }
            ASSERT(false);
        }
        INFO(n);
        INFO(m);
        cin.close();
    }

    void add_edge(int a, int b, double p)
    {
        probT[a].push_back(p);
        neighbors[a].push_back(b);
        inDeg[b]++;
        outDeg[a]++;
    }

    vector<bool> hasnode;
    void readGraph()
    {
        FILE *fin = fopen((graph_file).c_str(), "r");
        // ASSERT(fin != false);
        int readCnt = 0;
        for (int i = 0; i < m; i++)
        {
            readCnt ++;
            int a, b;
            double p;
            int c = fscanf(fin, "%d%d%lf", &a, &b, &p);
            ASSERTT(c == 3, a, b, p, c);

            //TRACE_LINE(a, b);
            ASSERT( a < n );
            ASSERT( b < n );
            hasnode[a] = true;
            hasnode[b] = true;

            add_edge(a, b, p);
        }
        TRACE_LINE_END();
        int s = 0;
        for (int i = 0; i < n; i++)
            if (hasnode[i])
                s++;
        ASSERT(readCnt == m);
        ASSERT(s == n);
        fclose(fin);
    }

    void readGraphBin()
    {
        string graph_file_bin = graph_file.substr(0, graph_file.size() - 3) + "bin";
        INFO(graph_file_bin);
        FILE *fin = fopen(graph_file_bin.c_str(), "rb");
        //fread(fin);
        struct stat filestatus;
        stat( graph_file_bin.c_str(), &filestatus );
        int64 sz = filestatus.st_size;
        char *buf = new char[sz];
        int64 sz2 = fread(buf, 1, sz, fin);
        INFO("fread finish", sz, sz2);
        ASSERT(sz == sz2);
        for (int64 i = 0; i < sz / 12; i++)
        {
            int a = ((int *)buf)[i * 3 + 0];
            int b = ((int *)buf)[i * 3 + 1];
            float p = ((float *)buf)[i * 3 + 2];
            //INFO(a,b,p);
            add_edge(a, b, p);
        }
        delete []buf;
        fclose(fin);
    }

    Graph(string folder, string graph_file): folder(folder), graph_file(graph_file)
    {
        readNM();

        //init vector
        FOR(i, n)
        {
            neighbors.push_back(vector<int>());
            hasnode.push_back(false);
            probT.push_back(vector<double>());
            outDeg.push_back(0);
            inDeg.push_back(0);
        }
        readGraph();
        
    }

};
double sqr(double t)
{
    return t * t;
}



