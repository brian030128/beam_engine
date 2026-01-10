"""KVTreeNode structure for FastTree kernel."""

class KVTreeNode:
    """
    Tree node structure compatible with FastTree kernel.

    Attributes:
        parent: Parent node ID (-1 for root)
        id: Unique node identifier
        seqlen: Total number of tokens from root to this node (cumulative)
        num_children: Number of child nodes
        requests: List of request IDs (candidate indices) using this node
    """
    def __init__(self):
        self.parent: int = -1
        self.id: int = 0
        self.seqlen: int = 0
        self.num_children: int = 0
        self.requests: list = []
