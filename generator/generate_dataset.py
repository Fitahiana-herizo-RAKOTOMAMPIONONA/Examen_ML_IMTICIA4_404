import csv
import os
from typing import List, Optional, Set, Tuple

X: int = 1
O: int = -1
EMPTY: int = 0

WIN_LINES: List[Tuple[int, int, int]] = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


class Node:
    def __init__(self, board: List[int]):
        self.board = list(board)
        self.winner = self._check_winner()

    def _check_winner(self) -> Optional[int]:
        for a, b, c in WIN_LINES:
            if self.board[a] == self.board[b] == self.board[c] != EMPTY:
                return self.board[a]
        return None

    def is_full(self) -> bool:
        return all(cell != EMPTY for cell in self.board)

    def get_turn(self) -> int:
        x_count = self.board.count(X)
        o_count = self.board.count(O)
        return X if x_count == o_count else O

    def getsuc(self) -> List['Node']:
        if self.winner is not None or self.is_full():
            return []

        successors = []
        player = self.get_turn()
        for i in range(9):
            if self.board[i] == EMPTY:
                new_board = list(self.board)
                new_board[i] = player
                successors.append(Node(new_board))
        return successors

    def is_valid(self) -> bool:
        x_count = self.board.count(X)
        o_count = self.board.count(O)

        if not (x_count == o_count or x_count == o_count + 1):
            return False

        x_wins = self._check_winner() == X
        o_wins = self._check_winner() == O
        if x_wins and o_wins:
            return False

        if x_wins and x_count != o_count + 1:
            return False
        if o_wins and x_count != o_count:
            return False

        return True

    def encode(self) -> List[int]:
        features = []
        for cell in self.board:
            features.append(1 if cell == X else 0)
            features.append(1 if cell == O else 0)
        return features

    def get_minimax_score(self, alpha: int = -2, beta: int = 2) -> int:
        if self.winner == X:
            return 1
        if self.winner == O:
            return -1
        if self.is_full():
            return 0

        turn = self.get_turn()
        successors = self.getsuc()

        if turn == X:
            best = -2
            for succ in successors:
                score = succ.get_minimax_score(alpha, beta)
                best = max(best, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return best
        else:
            best = 2
            for succ in successors:
                score = succ.get_minimax_score(alpha, beta)
                best = min(best, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return best


def generate_dataset():
    dataset = []
    seen = set()

    def explore(node: Node):
        board_tuple = tuple(node.board)
        if board_tuple in seen:
            return
        seen.add(board_tuple)

        if not node.is_valid():
            return

        if node.get_turn() == X and node.winner is None and not node.is_full():
            score = node.get_minimax_score()
            x_wins = 1 if score == 1 else 0
            is_draw = 1 if score == 0 else 0
            
            features = node.encode()
            dataset.append(features + [x_wins, is_draw])

        for succ in node.getsuc():
            explore(succ)

    root = Node([EMPTY] * 9)
    explore(root)
    return dataset


def main():
    print("Génération du dataset...")
    
    dataset = generate_dataset()

    headers = []
    for i in range(9):
        headers.extend([f"c{i}_x", f"c{i}_O"])
    headers.extend(["x_wins", "is_draw"])

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ressources")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dataset.csv")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(dataset)

    print(f"Dataset régénéré avec succès ({len(dataset)} états) !")
    print(f"Fichier : {output_path}")


if __name__ == "__main__":
    main()
