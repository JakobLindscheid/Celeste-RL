class Node:
    def __init__(self, state, action_taken=None, parent=None):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.g_cost = 0
        self.h_cost = 0
        self.f_cost = 0

    def __eq__(self, other):
        return set(self.state) == set(other.state)

    def __hash__(self):
        return hash(tuple([tuple(self.state), self.f_cost]))

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def __repr__(self):
        return f'Node(g_cost={self.g_cost}, h_cost={self.h_cost}, f_cost={self.f_cost})'
