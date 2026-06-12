import tkinter as tk
from engine import Engine
from input_manager import InputManager
from app import RasterizerApp

def main():
    root = tk.Tk()
    
    input_manager = InputManager()
    engine = Engine(screen_size=250)
    
    app = RasterizerApp(root, engine, input_manager)
    
    # Start App
    root.mainloop()

if __name__ == "__main__":
    main()