class Solution:
    """
    Generic base class for a solution, defining common attributes.
    """
    def __init__(self):
        self.cost: float = float('inf')
        self.is_feasible: bool = False

    def copy(self):
        """
        This method must be implemented by child classes
        to ensure a correct deep copy.
        """
        raise NotImplementedError