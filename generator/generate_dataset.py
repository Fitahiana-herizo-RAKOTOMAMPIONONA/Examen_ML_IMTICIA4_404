#!/usr/bin/env python3
"""
Générateur de dataset Morpion (Tic-Tac-Toe)
Utilise l'algorithme Minimax avec élagage Alpha-Bêta pour parcourir
et labelliser tous les états valides du jeu.

Le dataset contient uniquement les états où c'est au tour de X de jouer.
Features : c0_x, c0_o, c1_x, c1_o, ..., c8_x, c8_o (18 colonnes binaires)
Targets  : x_wins, is_draw
"""

import csv
import os
import sys

# Constantes
X = 1
O = -1
EMPTY = 0

# Lignes gagnantes (indices du plateau 3x3)
WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # lignes
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # colonnes
    (0, 4, 8), (2, 4, 6),             # diagonales
]


def check_winner(board):
    """Retourne X, O ou None."""
    for a, b, c in WIN_LINES:
        if board[a] == board[b] == board[c] != EMPTY:
            return board[a]
    return None


def is_full(board):
    return all(cell != EMPTY for cell in board)


def get_turn(board):
    """Retourne le joueur dont c'est le tour (X commence)."""
    x_count = board.count(X)
    o_count = board.count(O)
    return X if x_count == o_count else O


def minimax(board, is_maximizing, alpha, beta):
    """
    Minimax avec élagage Alpha-Bêta.
    Retourne le score optimal : +1 (X gagne), -1 (O gagne), 0 (nul).
    """
    winner = check_winner(board)
    if winner == X:
        return 1
    if winner == O:
        return -1
    if is_full(board):
        return 0

    if is_maximizing:
        best = -2
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = X
                score = minimax(board, False, alpha, beta)
                board[i] = EMPTY
                best = max(best, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
        return best
    else:
        best = 2
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = O
                score = minimax(board, True, alpha, beta)
                board[i] = EMPTY
                best = min(best, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
        return best


def encode_board(board):
    """Encode le plateau en 18 features binaires."""
    features = []
    for cell in board:
        features.append(1 if cell == X else 0)   # ci_x
        features.append(1 if cell == O else 0)    # ci_o
    return features


def is_valid_state(board):
    """Vérifie si un état de plateau est valide (atteignable par un jeu normal)."""
    x_count = board.count(X)
    o_count = board.count(O)

    # X commence, donc x_count == o_count ou x_count == o_count + 1
    if not (x_count == o_count or x_count == o_count + 1):
        return False

    x_wins = check_winner(board) == X
    o_wins = check_winner(board) == O

    # Les deux ne peuvent pas gagner
    if x_wins and o_wins:
        return False

    # Si X a gagné, O n'a pas pu jouer après
    if x_wins and x_count != o_count + 1:
        return False

    # Si O a gagné, X n'a pas pu jouer après
    if o_wins and x_count != o_count:
        return False

    return True


def generate_all_states():
    """
    Génère tous les états valides où c'est au tour de X de jouer,
    et les labellise avec le résultat en jeu parfait.
    """
    dataset = []
    seen = set()

    def explore(board):
        board_tuple = tuple(board)

        # Éviter les doublons
        if board_tuple in seen:
            return
        seen.add(board_tuple)

        # Vérifier la validité
        if not is_valid_state(board):
            return

        # Si la partie est terminée, ne pas l'ajouter
        if check_winner(board) is not None or is_full(board):
            return

        turn = get_turn(board)

        # On ne garde que les états où X joue
        if turn == X:
            # Évaluer par minimax en jeu parfait
            score = minimax(board, True, -2, 2)
            x_wins = 1 if score == 1 else 0
            is_draw = 1 if score == 0 else 0

            features = encode_board(board)
            dataset.append(features + [x_wins, is_draw])

        # Explorer les coups suivants
        player = turn
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = player
                explore(board)
                board[i] = EMPTY

    # Commencer avec un plateau vide
    initial_board = [EMPTY] * 9
    explore(initial_board)

    return dataset


def main():
    print("🎮 Génération du dataset Morpion...")
    print("   Algorithme: Minimax avec élagage Alpha-Bêta")
    print()

    dataset = generate_all_states()

    # En-têtes
    headers = []
    for i in range(9):
        headers.extend([f"c{i}_x", f"c{i}_o"])
    headers.extend(["x_wins", "is_draw"])

    # Créer le dossier ressources s'il n'existe pas
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ressources")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dataset.csv")

    # Écrire le CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(dataset)

    print(f"✅ Dataset généré avec succès !")
    print(f"   📁 Fichier : {output_path}")
    print(f"   📊 Nombre d'états : {len(dataset)}")
    print(f"   📐 Features : 18 (c0_x, c0_o, ..., c8_x, c8_o)")
    print(f"   🎯 Targets : x_wins, is_draw")

    # Stats rapides
    x_wins_count = sum(1 for row in dataset if row[-2] == 1)
    draws_count = sum(1 for row in dataset if row[-1] == 1)
    o_wins_count = len(dataset) - x_wins_count - draws_count
    print(f"\n   Distribution :")
    print(f"     X gagne : {x_wins_count} ({100*x_wins_count/len(dataset):.1f}%)")
    print(f"     Nul     : {draws_count} ({100*draws_count/len(dataset):.1f}%)")
    print(f"     O gagne : {o_wins_count} ({100*o_wins_count/len(dataset):.1f}%)")


if __name__ == "__main__":
    main()
