import gurobipy

try:
        gurobipy.Model()
        print("GUROBI LICENSE is available")
except Exception as e:
        assert False, e
