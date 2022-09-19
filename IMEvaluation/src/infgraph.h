#define SEED 1993
typedef pair<double, int> dipair;

#include "iheap.h"
#include <queue>
#include <utility>

class InfGraph : public Graph
{
private:
    vector<bool> visit;
    vector<int> visit_mark;

public:
    vector<vector<int>> hyperG;
    vector<vector<int>> hyperGT;
    int seed_random = 79340556;

    InfGraph(string folder, string graph_file) : Graph(folder, graph_file)
    {
        sfmt_init_gen_rand(&sfmtSeed, seed_random);
        init_hyper_graph();
        visit = vector<bool>(n);
        visit_mark = vector<int>(n);
    }

    void init_hyper_graph()
    {
        hyperG.clear();
        for (int i = 0; i < n; i++)
            hyperG.push_back(vector<int>());
        hyperGT.clear();
    }
    void build_hyper_graph_r(const Argument &arg)
    {
        int64 R = arg.size;
        if (R > INT_MAX)
        {
            cout << "Error:R too large" << endl;
            exit(1);
        }

        int prevSize = hyperGT.size();
        while ((int)hyperGT.size() <= R)
            hyperGT.push_back(vector<int>());

        vector<int> random_number;
        for (int i = 0; i < R; i++)
        {
            random_number.push_back(sfmt_genrand_uint32(&sfmtSeed) % n);
        }

        //trying BFS start from same node

        for (int i = prevSize; i < R; i++)
        {
#ifdef CONTINUOUS
            BuildHypergraphNode(random_number[i], i, arg);
#endif
#ifdef DISCRETE
            BuildHypergraphNode(random_number[i], i);
#endif
        }

        int totAddedElement = 0;
        for (int i = prevSize; i < R; i++)
        {
            for (int t : hyperGT[i])
            {
                hyperG[t].push_back(i);
                //hyperG.addElement(t, i);
                totAddedElement++;
            }
        }
    }

#ifdef DISCRETE
#include "discrete_rrset.h"
#endif
#ifdef CONTINUOUS
#include "continuous_rrset.h"
#endif

    //return the number of edges visited
    deque<int> q;
    sfmt_t sfmtSeed;

    double InfluenceHyperGraph(vector<int> seedSet, Argument arg)
    {
        set<int> s;
        TRACE(seedSet);
        for (auto t : seedSet)
        {
            for (auto tt : hyperG[t])
            {
                s.insert(tt);
            }
        }
        double inf = (double)s.size() / arg.size;
        return inf;
    }
};
