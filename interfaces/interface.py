import customtkinter as ctk
from tkinter import messagebox
import os
from PIL import Image

# Configuration globale
ctk.set_appearance_mode("light") 
ctk.set_default_color_theme("green") 

class InterfaceJeu:
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        
        # --- LOGIQUE DU JEU ---
        self.joueur = 'X'
        self.couleurs = {'X': "#198639", 'O': "#3498db"} # Vert ISPM pour X, Bleu pour O
        self.plateau = ['' for _ in range(9)]
        self.boutons = []

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
        chemin_logo = os.path.join("logo", "logoispm.png")
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

    def afficher_fin_match(self, message):
        """Overlay de fin de partie avec logo ISPM"""
        # Création de l'écran blanc par-dessus
        self.overlay = ctk.CTkFrame(self.root, fg_color="#FFFFFF", corner_radius=0)
        self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Conteneur pour centrer les éléments
        content = ctk.CTkFrame(self.overlay, fg_color="transparent")
        content.place(relx=0.5, rely=0.5, anchor="center")

        # AFFICHAGE DU LOGO ISPM
        if hasattr(self, 'logo_img'):
            self.logo_img.configure(size=(240, 300)) 
            logo_fin = ctk.CTkLabel(content, image=self.logo_img, text="")
            logo_fin.pack(pady=(0, 0))

        # Icône (Trophée ou Main)
        icone = "🏆" if "VICTOIRE" in message else "🤝"
        label_icon = ctk.CTkLabel(content, text=icone, font=ctk.CTkFont(size=80))
        label_icon.pack(pady=0)

        # Message (Victoire ou Nul)
        ctk.CTkLabel(
            content, 
            text=message, 
            font=ctk.CTkFont(family="Orbitron", size=26, weight="bold"),
            text_color="#198639" if "VICTOIRE" in message else "#3498db"
        ).pack(pady=20)

        # Bouton Rejouer
        ctk.CTkButton(
            content, text="REJOUER", font=ctk.CTkFont(size=16, weight="bold"),
            width=220, height=50, corner_radius=15,
            fg_color="#198639", hover_color="#146c2e",
            command=self.fermer_overlay
        ).pack(pady=10)

        # Bouton Menu
        ctk.CTkButton(
            content, text="MENU PRINCIPAL", font=ctk.CTkFont(size=16, weight="bold"),
            width=220, height=50, corner_radius=15,
            fg_color="#198639", hover_color="#146c2e",
            command=self.retour_au_menu
        ).pack(pady=10)

    def clic(self, i):
        if self.plateau[i] == '':
            self.plateau[i] = self.joueur
            self.boutons[i].configure(
                text=self.joueur, 
                fg_color=self.couleurs[self.joueur],
                border_color=self.couleurs[self.joueur]
            )
            
            victoire = self.verifier_victoire()
            if victoire:
                for idx in victoire:
                    self.boutons[idx].configure(text_color="black")
                self.root.after(400, lambda: self.afficher_fin_match(f"VICTOIRE DE {self.joueur}"))
            elif '' not in self.plateau:
                self.afficher_fin_match("MATCH NUL !")
            else:
                self.joueur = 'O' if self.joueur == 'X' else 'X'
                self.actualiser_style_tour()

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