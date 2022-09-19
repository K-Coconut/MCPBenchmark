#define SEED 1993
typedef pair<double, int> dipair;

#include "iheap.h"
#include <queue>
#include <utility>

struct CompareBySecond
{
    bool operator()(pair<int, int> a, pair<int, int> b)
    {
        return a.second < b.second;
    }
};

class InfGraph : public Graph
{
private:
    vector<bool> visit;
    vector<int> visit_mark;

public:
    vector<vector<int>> hyperG;
    vector<vector<int>> hyperGT;

    InfGraph(string folder, string graph_file, int num_k, int seed_random) : Graph(folder, graph_file, num_k)
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
    void build_hyper_graph_r(int64 R, const Argument &arg)
    {
        if (R > INT_MAX)
        {
            cout << "Error:R too large" << endl;
            exit(1);
        }
        // INFO("build_hyper_graph_r", R);

        int prevSize = hyperGT.size();
        while ((int)hyperGT.size() <= R)
            hyperGT.push_back(vector<int>());

        vector<int> random_number;
        for (int i = 0; i < R; i++)
        {
            random_number.push_back(sfmt_genrand_uint32(&sfmtSeed) % n);
        }

        // trying BFS start from same node

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
                // hyperG.addElement(t, i);
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

    // return the number of edges visited
    deque<int> q;
    sfmt_t sfmtSeed;
    vector<int> seedSet;
    map<int, int> dict_node_id_gain;

    double build_seedset(int k)
    {
        priority_queue<pair<int, int>, vector<pair<int, int>>, CompareBySecond> heap;
        vector<int> coverage(n, 0);
        for (int i = 0; i < n; i++)
        {
            pair<int, int> tep(make_pair(i, (int)hyperG[i].size()));
            heap.push(tep);
            coverage[i] = (int)hyperG[i].size();
        }

        int maxInd;

        long long influence = 0;
        long long numEdge = hyperGT.size();

        // check if an edge is removed
        vector<bool> edgeMark(numEdge, false);
        // check if an node is remained in the heap
        vector<bool> nodeMark(n + 1, true);

        seedSet.clear();
        while ((int)seedSet.size() < k)
        {
            pair<int, int> ele = heap.top();
            heap.pop();
            if (ele.second > coverage[ele.first])
            {
                ele.second = coverage[ele.first];
                heap.push(ele);
                continue;
            }

            maxInd = ele.first;
            vector<int> e = hyperG[maxInd];
            influence += coverage[maxInd];
            seedSet.push_back(maxInd);
            dict_node_id_gain[maxInd] = ele.second;
            nodeMark[maxInd] = false;

            for (unsigned int j = 0; j < e.size(); ++j)
            {
                if (edgeMark[e[j]])
                    continue;

                vector<int> nList = hyperGT[e[j]];
                for (unsigned int l = 0; l < nList.size(); ++l)
                {
                    if (nodeMark[nList[l]])
                        coverage[nList[l]]--;
                }
                edgeMark[e[j]] = true;
            }
        }
        TRACE(seedSet);
        return 1.0 * influence / hyperGT.size();
    }
};
