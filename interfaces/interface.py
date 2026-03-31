import customtkinter as ctk
from tkinter import messagebox
import os
import json
import math
import time
from PIL import Image

# Configuration globale
ctk.set_appearance_mode("light") 
ctk.set_default_color_theme("green") 

class AI:
    def __init__(self, models_path):
        self.data = None
        try:
            if os.path.exists(models_path):
                with open(models_path, "r") as f:
                    self.data = json.load(f)
            else:
                print(f"Fichier modèle introuvable : {models_path}")
        except Exception as e:
            print(f"Erreur chargement modèles: {e}")

    def predict_x_wins(self, board):
        if not self.data: return 0.5
        
        # Encodage (18 features)
        features = []
        for cell in board:
            features.append(1 if cell == 'X' else 0)
            features.append(1 if cell == 'O' else 0)
            
        # Normalisation
        scaler = self.data["scaler"]
        scaled = [(features[i] - scaler["mean"][i]) / scaler["scale"][i] for i in range(18)]
        
        # Logistique
        model = self.data["lr_xwins"]
        z = sum(scaled[i] * model["coef"][i] for i in range(18)) + model["intercept"]
        return 1 / (1 + math.exp(-z))

    def get_best_move_ml(self, board):
        meilleur_score = 2.0 # On veut minimiser P(X wins) car on est 'O'
        meilleur_coup = -1
        
        for i in range(9):
            if board[i] == '':
                board_temp = list(board)
                board_temp[i] = 'O'
                score = self.predict_x_wins(board_temp)
                if score < meilleur_score:
                    meilleur_score = score
                    meilleur_coup = i
        return meilleur_coup

    def minimax(self, board, depth, is_maximizing):
        # Evaluation
        victoire = self.verifier_victoire_statique(board)
        if victoire == 'O': return 10 - depth
        if victoire == 'X': return depth - 10
        if '' not in board: return 0

        if is_maximizing:
            best = -100
            for i in range(9):
                if board[i] == '':
                    board[i] = 'O'
                    best = max(best, self.minimax(board, depth + 1, False))
                    board[i] = ''
            return best
        else:
            best = 100
            for i in range(9):
                if board[i] == '':
                    board[i] = 'X'
                    best = min(best, self.minimax(board, depth + 1, True))
                    board[i] = ''
            return best

    def get_best_move_hybride(self, board):
        best_val = -100
        best_move = -1
        board_copy = list(board)
        for i in range(9):
            if board_copy[i] == '':
                board_copy[i] = 'O'
                move_val = self.minimax(board_copy, 0, False)
                board_copy[i] = ''
                if move_val > best_val:
                    best_move = i
                    best_val = move_val
        return best_move

    def verifier_victoire_statique(self, plateau):
        v = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for a, b, c in v:
            if plateau[a] == plateau[b] == plateau[c] != '':
                return plateau[a]
        return None

class InterfaceJeu:
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        
        # --- LOGIQUE DU JEU ---
        self.joueur = 'X'
        self.couleurs = {'X': "#198639", 'O': "#3498db"} # Vert ISPM pour X, Bleu pour O
        self.plateau = ['' for _ in range(9)]
        self.boutons = []
        
        # Initialisation IA
        script_dir = os.path.dirname(__file__)
        models_path = os.path.join(script_dir, "public", "models.json")
        self.ia = AI(models_path)

        # --- CONFIGURATION FENÊTRE ---
        for widget in self.root.winfo_children():
            widget.destroy()
            
        self.root.title(f"Morpion 404 - {mode}")
        self.root.geometry("450x650")
        self.root.configure(fg_color="#FFFFFF") 

        # --- HEADER ---
        self.header = ctk.CTkFrame(self.root, fg_color="transparent")
        self.header.pack(pady=20, padx=20, fill="x")

        self.titre = ctk.CTkLabel(
            self.header, 
            text="TOUR DU JOUEUR X", 
            font=ctk.CTkFont(family="Orbitron", size=20, weight="bold"),
            text_color=self.couleurs['X']
        )
        self.titre.pack(side="left", padx=10)

        # Chargement du Logo ISPM
        chemin_logo = os.path.join(script_dir, "logo", "logoispm.png")
        try:
            img_data = Image.open(chemin_logo)
            self.logo_img = ctk.CTkImage(light_image=img_data, dark_image=img_data, size=(80, 100))
            self.logo_label = ctk.CTkLabel(self.header, image=self.logo_img, text="")
            self.logo_label.pack(side="right", padx=10)
        except Exception as e:
            print(f"Erreur logo: {e}")

        # --- GRILLE DE JEU ---
        self.main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main_container.pack(expand=True)

        for i in range(9):
            btn = ctk.CTkButton(
                self.main_container, 
                text='', 
                font=ctk.CTkFont(size=35, weight="bold"),
                width=110, 
                height=110,
                fg_color="#F0F0F0", 
                hover_color=self.couleurs['X'],
                text_color="white",
                command=lambda i=i: self.clic(i)
            )
            btn.grid(row=i//3, column=i%3, padx=6, pady=6)
            self.boutons.append(btn)

    def actualiser_style_tour(self):
        couleur = self.couleurs[self.joueur]
        self.titre.configure(text=f"TOUR DU JOUEUR {self.joueur}", text_color=couleur)
        for btn in self.boutons:
            if btn.cget("text") == '':
                btn.configure(hover_color=couleur)

    def clic(self, i):
        if self.plateau[i] == '' and self.joueur == 'X':
            self.jouer_tour(i)
            
            # Tour de l'IA (si pas mode Human)
            if self.mode != "Human" and '' in self.plateau and not self.verifier_victoire():
                self.root.after(600, self.jouer_ia)
        elif self.plateau[i] == '' and self.mode == "Human" and self.joueur == 'O':
            self.jouer_tour(i)

    def jouer_tour(self, i):
        if self.plateau[i] != '': return

        self.plateau[i] = self.joueur
        self.boutons[i].configure(
            text=self.joueur, 
            fg_color=self.couleurs[self.joueur],
            border_color=self.couleurs[self.joueur]
        )
        
        victoire = self.verifier_victoire()
        if victoire:
            for idx in victoire:
                self.boutons[idx].configure(text_color="yellow") 
            self.root.after(400, lambda: self.afficher_fin_match(f"VICTOIRE DE {self.joueur}"))
        elif '' not in self.plateau:
            self.afficher_fin_match("MATCH NUL !")
        else:
            self.joueur = 'O' if self.joueur == 'X' else 'X'
            self.actualiser_style_tour()

    def jouer_ia(self):
        if self.mode == "ML":
            move = self.ia.get_best_move_ml(self.plateau)
        else: # Hybride
            move = self.ia.get_best_move_hybride(self.plateau)
            
        if move != -1:
            self.jouer_tour(move)

    def afficher_fin_match(self, message):
        self.overlay = ctk.CTkFrame(self.root, fg_color="#FFFFFF", corner_radius=0)
        self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)

        content = ctk.CTkFrame(self.overlay, fg_color="transparent")
        content.place(relx=0.5, rely=0.5, anchor="center")

        if hasattr(self, 'logo_img'):
            self.logo_img.configure(size=(240, 300)) 
            logo_fin = ctk.CTkLabel(content, image=self.logo_img, text="")
            logo_fin.pack(pady=(0, 0))

        icone = "🏆" if "VICTOIRE" in message else "🤝"
        label_icon = ctk.CTkLabel(content, text=icone, font=ctk.CTkFont(size=80))
        label_icon.pack(pady=0)

        ctk.CTkLabel(
            content, 
            text=message, 
            font=ctk.CTkFont(family="Orbitron", size=26, weight="bold"),
            text_color="#198639" if "VICTOIRE" in message else "#3498db"
        ).pack(pady=20)

        ctk.CTkButton(
            content, text="REJOUER", font=ctk.CTkFont(size=16, weight="bold"),
            width=220, height=50, corner_radius=15,
            fg_color="#198639", hover_color="#146c2e",
            command=self.fermer_overlay
        ).pack(pady=10)

        ctk.CTkButton(
            content, text="MENU PRINCIPAL", font=ctk.CTkFont(size=16, weight="bold"),
            width=220, height=50, corner_radius=15,
            fg_color="#198639", hover_color="#146c2e",
            command=self.retour_au_menu
        ).pack(pady=10)

    def verifier_victoire(self):
        v = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for a, b, c in v:
            if self.plateau[a] == self.plateau[b] == self.plateau[c] != '':
                return (a, b, c)
        return False

    def reset(self):
        self.plateau = ['' for _ in range(9)]
        self.joueur = 'X'
        for b in self.boutons:
            b.configure(text='', fg_color="#F0F0F0", border_color="#E0E0E0")
        self.actualiser_style_tour()

    def fermer_overlay(self):
        if hasattr(self, 'overlay'):
            self.overlay.destroy()
        self.reset()

    def retour_au_menu(self):
        from level import LevelMenu
        LevelMenu(self.root)