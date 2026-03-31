import customtkinter as ctk
from interface import InterfaceJeu
import os
from PIL import Image

# Forcer le mode clair et le fond blanc
ctk.set_appearance_mode("light") 
ctk.set_default_color_theme("green")

class LevelMenu:
    def __init__(self, root):
        self.root = root
        
        # Nettoyage
        for widget in self.root.winfo_children():
            widget.destroy()

        self.root.title("Morpion 404 - Configuration")
        self.root.geometry("450x600")
        
        # --- FORCER LE FOND BLANC ICI ---
        self.root.configure(fg_color="white")

        # --- HEADER ---
        self.header = ctk.CTkFrame(self.root, fg_color="white") # Fond blanc
        self.header.pack(pady=20, padx=20, fill="x")

        # Titre
        self.titre = ctk.CTkLabel(
            self.header, 
            text="CHOISIR MODE", 
            font=ctk.CTkFont(family="Orbitron", size=22, weight="bold"),
            text_color="#198639"
        )
        self.titre.pack(side="left", padx=10)

        # --- LOGO ---
        # Utilisation d'un chemin absolu pour éviter les erreurs
        script_dir = os.path.dirname(__file__) 
        chemin_logo = os.path.join(script_dir, "logo", "logoispm.png")

        if os.path.exists(chemin_logo):
            try:
                img_data = Image.open(chemin_logo)
                # On définit l'image pour le mode light et dark
                self.logo_img = ctk.CTkImage(
                    light_image=img_data, 
                    dark_image=img_data, 
                    size=(105, 125)
                )
                self.logo_label = ctk.CTkLabel(self.header, image=self.logo_img, text="")
                self.logo_label.pack(side="right", padx=10)
            except Exception as e:
                print(f"Erreur lors de l'ouverture de l'image: {e}")
        else:
            print(f"Fichier introuvable : {chemin_logo}")

        # --- RESTE DU CODE (BOUTONS) ---
        self.frame_buttons = ctk.CTkFrame(self.root, fg_color="white")
        self.frame_buttons.pack(expand=True, fill="both", padx=50)

        btn_kwargs = {
            "font": ctk.CTkFont(size=16, weight="bold"),
            "height": 55,
            "corner_radius": 10,
            "fg_color": "#198639",
            "hover_color": "#1e9e43",
            "text_color": "white"
        }

        self.btn_human = ctk.CTkButton(self.frame_buttons, text="VS HUMAN", 
                                      command=lambda: InterfaceJeu(self.root, "Human"), **btn_kwargs)
        self.btn_human.pack(pady=12, fill="x")

        self.btn_ml = ctk.CTkButton(self.frame_buttons, text="VS IA (ML)", 
                                   command=lambda: InterfaceJeu(self.root, "ML"), **btn_kwargs)
        self.btn_ml.pack(pady=12, fill="x")

        self.btn_hybride = ctk.CTkButton(self.frame_buttons, text="VS IA (HYBRIDE)", 
                                        command=lambda: InterfaceJeu(self.root, "Hybride"), **btn_kwargs)
        self.btn_hybride.pack(pady=12, fill="x")

      

if __name__ == "__main__":
    root = ctk.CTk()
    app = LevelMenu(root)
    root.mainloop()