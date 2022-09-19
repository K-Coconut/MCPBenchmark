import ctypes
import os
import sys

class MvcLib(object):

    def __init__(self, args):
        dir_path = os.path.dirname(os.path.realpath(__file__))        
        self.lib = ctypes.CDLL('%s/build/dll/libmvc.so' % dir_path)

        self.lib.Fit.restype = ctypes.c_double
        self.lib.Test.restype = ctypes.c_double
        self.lib.GetSol.restype = ctypes.c_double
        self.lib.GetBudgetedSol.restype = ctypes.c_double
        arr = (ctypes.c_char_p * len(args))()
        for i in range(len(args)):
            arr[i] = args[i].encode()
        self.lib.Init(len(arr), arr)
        self.ngraph_train = 0
        self.ngraph_test = 0

    def __CtypeNetworkX(self, g):
        edges = g.edges()
        e_list_from = (ctypes.c_int * len(edges))()
        e_list_to = (ctypes.c_int * len(edges))()

        if len(edges):
            a, b = zip(*edges)        
            e_list_from[:] = a
            e_list_to[:] = b

        return (len(g.nodes()), len(edges), ctypes.cast(e_list_from, ctypes.c_void_p), ctypes.cast(e_list_to, ctypes.c_void_p)) 

    def TakeSnapshot(self):
        self.lib.UpdateSnapshot()

    def ClearTrainGraphs(self):
        self.ngraph_train = 0
        self.lib.ClearTrainGraphs()

    def InsertGraph(self, g, is_test):
        n_nodes, n_edges, e_froms, e_tos = self.__CtypeNetworkX(g)
        if is_test:
            t = self.ngraph_test
            self.ngraph_test += 1
        else:
            t = self.ngraph_train
            self.ngraph_train += 1

        self.lib.InsertGraph(is_test, t, n_nodes, n_edges, e_froms, e_tos)
    
    def LoadModel(self, path_to_model):
        self.lib.LoadModel(path_to_model.encode())

    def SaveModel(self, path_to_model):
        self.lib.SaveModel(path_to_model.encode())

    def GetSol(self, gid, maxn):
        sol = (ctypes.c_int * (maxn + 10))()
        val = self.lib.GetSol(gid, sol)
        return val, sol

    def GetBudgetedSol(self, gid, maxn, budget):
        sol = (ctypes.c_int * (maxn + 10))()
        val = self.lib.GetBudgetedSol(gid, sol, budget)
        return val, sol

if __name__ == '__main__':
    f = MvcLib(sys.argv)
