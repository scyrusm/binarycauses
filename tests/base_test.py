### A basic, and hardly robust, test
n_nodes = 4
T = 10000
art = Artificial()
art.generate(n_nodes, T)
empir = EmpiricalCausalAdjacencySolver()
empir.solve(art.X, art.Y)
assert art.unary_adjacencies.shape == empir.empirical_unary_adjacencies.shape
assert len(art.binary_adjencies.keys()) == len(empir.empirical_binary_adjacencies.keys())
