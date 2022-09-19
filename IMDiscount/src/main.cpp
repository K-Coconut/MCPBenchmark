#include "graph.h"
#include <queue>
#include <sstream>
#include <time.h>
#include "memoryusage.h"

using namespace std;
namespace fs = std::experimental::filesystem;

class Argument
{
public:
   string algorithm;
   int k;
   string graph_folder;
   string output_prefix;
};

struct node
{
   int idx;
   float degree;
   node(int n, float v) : idx(n), degree(v) {}
   friend bool operator<(const struct node &n1, const struct node &n2);
};

struct cmp
{
   bool operator()(node a, node b)
   {
      if (a.degree != b.degree)
         return a.degree < b.degree;
      return a.idx > b.idx;
   }
};

int getMaxElement(vector<float> v)
{
   vector<float>::iterator result;
   result = max_element(v.begin(), v.end());
   int index = distance(v.begin(), result);
   return index;
}

vector<int> dDiscount(Argument &arg, Graph &g)
{
   vector<float> discount_degree;
   vector<int> t_v;
   vector<bool> visited;
   for (int i = 0; i < g.n; i++)
   {
      discount_degree.push_back(g.outDeg[i]);
      t_v.push_back(0);
      visited.push_back(false);
   }

   vector<int> S;
   for (int i = 0; i < arg.k; i++)
   {
      int u = getMaxElement(discount_degree);
      S.push_back(u);
      discount_degree[u] = -1;
      visited[u] = true;
      for (int ind = 0; ind < g.neighbors[u].size(); ind++)
      {
         int v = g.neighbors[u][ind];
         if (visited[v])
            continue;
         t_v[v] += 1;
         int dv = g.outDeg[v];
         int tv = t_v[v];
         float ddv = dv - 2 * tv - (dv - tv) * tv * g.probT[u][ind];
         discount_degree[v] = ddv;
      }
   }
   return S;
}

vector<int> singleDiscount(Argument &arg, Graph &g)
{
   vector<float> discount_degree;
   vector<bool> visited;
   for (int i = 0; i < g.n; i++)
   {
      discount_degree.push_back(g.outDeg[i]);
      visited.push_back(false);
   }

   vector<int> S;
   for (int i = 0; i < arg.k; i++)
   {
      int u = getMaxElement(discount_degree);
      S.push_back(u);
      discount_degree[u] = -1;
      visited[u] = true;
      for (int ind = 0; ind < g.neighbors[u].size(); ind++)
      {
         int v = g.neighbors[u][ind];
         if (visited[v])
            continue;

         int dv = g.outDeg[v];
         int ddv = dv - 1;
         discount_degree[v] = ddv;
      }
   }
   return S;
}

void writeResults(vector<int> S, Argument &arg, double total_time, double memory)
{
   string seed_file = arg.output_prefix + "_seeds.txt";
   string time_file = arg.output_prefix + "_time.txt";
   string mem_file = arg.output_prefix + "_memory.txt";
   ofstream outputFile;
   outputFile.open(seed_file.c_str());
   for (auto s : S)
   {
      outputFile << s << endl;
   }
   outputFile.close();

   outputFile.open(time_file.c_str());
   outputFile << total_time << endl;
   outputFile.close();

   outputFile.open(mem_file.c_str());
   outputFile << memory << endl;
   outputFile.close();
}

void run(int argn, char **argv)
{
   Argument arg;

   // default argument
   arg.algorithm = "";

   for (int i = 0; i < argn; i++)
   {
      if (argv[i] == string("-help"))
      {
         cout << "./discount -dataset *** -a algorithm DDiscount/SingleDiscount -k budget -num number of iterations" << endl;
         return;
      }
      if (argv[i] == string("-algorithm"))
         arg.algorithm = argv[i + 1];
      if (argv[i] == string("-k"))
         arg.k = atoi(argv[i + 1]);
      if (argv[i] == string("-graph_folder"))
         arg.graph_folder = argv[i + 1];
      if (argv[i] == string("-output_prefix"))
         arg.output_prefix = argv[i + 1];
   }
   ASSERT(arg.algorithm == "DDiscount" || arg.algorithm == "SingleDiscount");

   string graph_folder = arg.graph_folder;
   string graph_file = arg.graph_folder + "/edges.txt";
   Graph g(graph_folder, graph_file);
   vector<int> S;

   clock_t start_time = clock();
   if (arg.algorithm == "DDiscount")
   {
      S = dDiscount(arg, g);
   }
   else
   {
      S = singleDiscount(arg, g);
   }
   clock_t end_time = clock();
   double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
   cout << "Budget: " << arg.k << " Running time: " << total_time << " memory usage: " << disp_mem_usage("") << endl;
   writeResults(S, arg, total_time, disp_mem_usage(""));
}

int main(int argn, char **argv)
{

   run(argn, argv);
}
