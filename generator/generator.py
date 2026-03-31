class Node:

    ci_X: list[int]
    ci_O: list[int]

    def is_Win(data: list[int]) -> bool:
        for i in [0, 3, 6]:
            if ((data[i] + data[i+1] + data[i+2]) == 3):
                return True
        for i in [0, 1, 2]:
            if ((data[i] + data[i+3] + data[i+6]) == 3):
                return True 
        if ((data[0] + data[4] + data[8]) == 3):
            return True
        if ((data[2] + data[4] + data[6]) == 3):
            return True
        return False
    
    def getSuccessors() -> list[Node]:
        res : list[Node] = []
        for
        for i in range(8):
        return res
    
