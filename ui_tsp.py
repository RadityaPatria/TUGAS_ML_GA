import tkinter as tk
from tkinter import ttk
import threading
import pandas as pd
import os
from ga_tsp import genetic_algorithm

# Matplotlib untuk visual
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

DATASET_PATH = "data/medium.csv"

# =====================================
# LOAD DATASET
# =====================================
def load_dataset():
    df = pd.read_csv(DATASET_PATH, header=None)
    df.columns = ["X", "Y"]

    cities = []
    for i, row in df.iterrows():
        cities.append({
            "City": f"City {i+1}",
            "X": float(row["X"]),
            "Y": float(row["Y"])
        })
    return cities


# =====================================
# VISUALIZE TSP ROUTE
# =====================================
def draw_route(route):
    figure = Figure(figsize=(5, 4), dpi=100)
    subplot = figure.add_subplot(111)

    xs = [c["X"] for c in route] + [route[0]["X"]]
    ys = [c["Y"] for c in route] + [route[0]["Y"]]

    subplot.plot(xs, ys, marker="o")
    subplot.set_title("TSP Best Route Visualization")
    subplot.set_xlabel("X")
    subplot.set_ylabel("Y")

    # Render to Tkinter canvas
    canvas = FigureCanvasTkAgg(figure, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


# =====================================
# RUN GA INSIDE THREAD
# =====================================
def run_ga_thread():
    progress.start()
    status_label.config(text="Running Genetic Algorithm...")
    output.delete("1.0", tk.END)

    try:
        cities = load_dataset()
    except Exception as e:
        output.insert(tk.END, f"Dataset Error: {e}")
        progress.stop()
        return

    # Ambil input parameter
    pop_size = int(pop_size_entry.get())
    generations = int(gen_entry.get())
    mutation = float(mut_entry.get())
    crossover = float(cross_entry.get())

    # RUN GA
    best_route, best_distance = genetic_algorithm(
        cities,
        population_size=pop_size,
        generations=generations,
        mutation_rate=mutation,
        crossover_rate=crossover
    )

    # Tampilkan hasil
    output.insert(tk.END, "===== GA RESULT =====\n")
    output.insert(tk.END, f"Total Cities: {len(cities)}\n")
    output.insert(tk.END, f"Population Size: {pop_size}\n")
    output.insert(tk.END, f"Generations: {generations}\n")
    output.insert(tk.END, f"Mutation Rate: {mutation}\n")
    output.insert(tk.END, f"Crossover Rate: {crossover}\n\n")

    output.insert(tk.END, f"Best Distance: {best_distance:.4f}\n\n")
    output.insert(tk.END, "Best Route:\n")

    for city in best_route:
        output.insert(tk.END, f"{city['City']} -> ")
    output.insert(tk.END, best_route[0]["City"])

    # Gambar route di UI
    for widget in graph_frame.winfo_children():
        widget.destroy()
    draw_route(best_route)

    status_label.config(text="Completed!")
    progress.stop()


def start_ga():
    threading.Thread(target=run_ga_thread, daemon=True).start()


# =====================================
# UI SETUP
# =====================================
root = tk.Tk()
root.title("Genetic Algorithm – TSP Visual Solver")
root.geometry("1000x700")
root.configure(bg="#202020")

header = tk.Label(root, text="Genetic Algorithm – TSP Solver with Visualization",
                  font=("Segoe UI", 18, "bold"),
                  fg="white", bg="#202020")
header.pack(pady=10)


# ------------ PARAMETER FORM ------------
form = tk.Frame(root, bg="#2d2d2d", pady=10, padx=20)
form.pack(pady=10)

tk.Label(form, text="Population Size:", fg="white", bg="#2d2d2d").grid(row=0, column=0, sticky="w")
pop_size_entry = tk.Entry(form, width=10)
pop_size_entry.insert(0, "100")
pop_size_entry.grid(row=0, column=1, padx=10)

tk.Label(form, text="Generations:", fg="white", bg="#2d2d2d").grid(row=1, column=0, sticky="w")
gen_entry = tk.Entry(form, width=10)
gen_entry.insert(0, "300")
gen_entry.grid(row=1, column=1, padx=10)

tk.Label(form, text="Mutation Rate:", fg="white", bg="#2d2d2d").grid(row=2, column=0, sticky="w")
mut_entry = tk.Entry(form, width=10)
mut_entry.insert(0, "0.1")
mut_entry.grid(row=2, column=1, padx=10)

tk.Label(form, text="Crossover Rate:", fg="white", bg="#2d2d2d").grid(row=3, column=0, sticky="w")
cross_entry = tk.Entry(form, width=10)
cross_entry.insert(0, "0.8")
cross_entry.grid(row=3, column=1, padx=10)


# ------------ BUTTON ------------
run_btn = tk.Button(root, text="RUN GA", bg="#4CAF50", fg="white",
                    font=("Segoe UI", 12, "bold"),
                    width=20, command=start_ga)
run_btn.pack(pady=10)

status_label = tk.Label(root, text="Ready", fg="white", bg="#202020",
                        font=("Segoe UI", 12))
status_label.pack()

progress = ttk.Progressbar(root, mode="indeterminate", length=300)
progress.pack(pady=10)


# ------------ OUTPUT TEXT ------------
output = tk.Text(root, height=12, width=110,
                 bg="#1e1e1e", fg="white",
                 font=("Consolas", 10))
output.pack(pady=10)

# ------------ GRAPH VISUAL ------------
graph_frame = tk.Frame(root, bg="#202020")
graph_frame.pack(fill=tk.BOTH, expand=True)

root.mainloop()
