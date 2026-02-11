# core/graph.py

class ComputationGraph:
    """
    Maintains and traverses the computation graph formed by Tensor objects.
    """

    def __init__(self):
        self.nodes = []

    def register(self, tensor):
        """
        Registers a tensor node in the graph.
        """
        if tensor not in self.nodes:
            self.nodes.append(tensor)

    def build_topological_order(self, output_tensor):
        """
        Returns tensors in topological order using DFS.
        """
        visited = set()
        order = []

        def dfs(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    dfs(parent)
                order.append(node)

        dfs(output_tensor)
        return order

    def clear(self):
        """
        Clears the stored graph.
        """
        self.nodes.clear()
