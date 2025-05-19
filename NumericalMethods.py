# file: numerical_methods_gui.py

import tkinter as tk
from tkinter import ttk, messagebox
from sympy import symbols, sympify, lambdify, diff
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import re

messagebox.showinfo("Instructions: Input all the values and Choose a method.")
messagebox.showinfo("Note: Regula Falsi and Bisection needs the lower limit to" \
"be negative")
messagebox.showinfo("they can't be usable on a quadratic interval and above the degree")
messagebox.showinfo("that turns both of them to positive")

x = symbols('x')

def insert_implicit_multiplication(expr_str):
    expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)
    expr_str = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', expr_str)
    return expr_str

class RootFinderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Numerical Methods Root Finder")
        self.root.geometry("1000x700")

        self.function_expr = tk.StringVar()
        self.method = tk.StringVar()
        self.method.set("Graphical Method")

        self.param_a = tk.DoubleVar()
        self.param_b = tk.DoubleVar()
        self.delta_x = tk.DoubleVar(value=0.01)
        self.tolerance = tk.DoubleVar(value=0.001)
        self.max_iter = tk.IntVar(value=100)

        self.create_widgets()

    def create_widgets(self):
        input_frame = ttk.LabelFrame(self.root, text="Input")
        input_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(input_frame, text="f(x) =").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.function_expr, width=40).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Method:").grid(row=1, column=0, padx=5, pady=5)
        methods = ["Graphical Method", "Incremental Method", "Bisection Method",
                   "Regula Falsi Method", "Newton-Raphson Method", "Secant Method"]
        method_combo = ttk.Combobox(input_frame, textvariable=self.method, values=methods, width=30)
        method_combo.grid(row=1, column=1, padx=5, pady=5)
        method_combo.bind("<<ComboboxSelected>>", self.update_param_fields)

        ttk.Label(input_frame, text="a (lower bound):").grid(row=2, column=0, padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.param_a, width=10).grid(row=2, column=1, sticky='w', padx=5, pady=5)

        ttk.Label(input_frame, text="b (upper bound):").grid(row=3, column=0, padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.param_b, width=10).grid(row=3, column=1, sticky='w', padx=5, pady=5)

        ttk.Label(input_frame, text="Δx (step size):").grid(row=4, column=0, padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.delta_x, width=10).grid(row=4, column=1, sticky='w', padx=5, pady=5)

        ttk.Label(input_frame, text="Tolerance:").grid(row=5, column=0, padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.tolerance, width=10).grid(row=5, column=1, sticky='w', padx=5, pady=5)

        ttk.Label(input_frame, text="Max Iterations:").grid(row=6, column=0, padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.max_iter, width=10).grid(row=6, column=1, sticky='w', padx=5, pady=5)

        ttk.Button(input_frame, text="Solve", command=self.solve).grid(row=7, column=0, columnspan=2, pady=10)

        self.output_frame = ttk.LabelFrame(self.root, text="Output")
        self.output_frame.pack(fill="both", expand=True, padx=10, pady=5)

    def update_param_fields(self, event=None):
        pass

    def solve(self):
        func_str = self.function_expr.get()
        try:
            func_str = insert_implicit_multiplication(func_str)
            expr = sympify(func_str)
            func = lambdify(x, expr, 'numpy')
            method = self.method.get()

            match method:
                case "Graphical Method": self.plot_graph(func, expr)
                case "Incremental Method": self.run_incremental(func, expr)
                case "Bisection Method": self.run_bisection(func, expr)
                case "Regula Falsi Method": self.run_regula_falsi(func, expr)
                case "Newton-Raphson Method": self.run_newton_raphson(func, expr)
                case "Secant Method": self.run_secant(func, expr)
                case _: messagebox.showinfo("Notice", f"Method '{method}' not recognized.")

        except Exception as e:
            messagebox.showerror("Error", f"Invalid expression: {e}")

    def plot_graph(self, func, expr):
        self.clear_output()
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        x_vals = np.linspace(-10, 10, 400)
        y_vals = func(x_vals)

        ax.plot(x_vals, y_vals, label=f"$f(x) = {expr}$")
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_title("Function Graph")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.output_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def run_incremental(self, func, expr):
        self.clear_output()
        a, b, step = self.param_a.get(), self.param_b.get(), self.delta_x.get()
        table = self.create_table(["x0", "x1", "f(x0)", "f(x1)"])
        roots = []
        x0 = a

        while x0 < b:
            x1 = x0 + step
            fx0, fx1 = func(x0), func(x1)
            table.insert("", "end", values=(f"{x0:.6f}", f"{x1:.6f}", f"{fx0:.4e}", f"{fx1:.4e}"))
            if fx0 * fx1 < 0:
                roots.append((x0 + x1) / 2)
            x0 = x1

        table.pack(fill="x", pady=10)
        self.plot_roots(func, expr, a, b, roots)

    def run_bisection(self, func, expr):
        self.clear_output()
        a, b, tol, max_iter = self.param_a.get(), self.param_b.get(), self.tolerance.get(), self.max_iter.get()

        if func(a) * func(b) >= 0:
            messagebox.showerror("Error", "f(a) and f(b) must have opposite signs")
            return

        table = self.create_table(["Iter", "a", "b", "c", "f(c)"])
        for i in range(max_iter):
            c = (a + b) / 2
            fc = func(c)
            table.insert("", "end", values=(i+1, f"{a:.6f}", f"{b:.6f}", f"{c:.6f}", f"{fc:.6e}"))
            if abs(fc) < tol: break
            if func(a) * fc < 0: b = c
            else: a = c

        table.pack(fill="x", pady=10)
        self.plot_graph_with_root(func, expr, c)

    def run_regula_falsi(self, func, expr):
        self.clear_output()
        a, b, tol, max_iter = self.param_a.get(), self.param_b.get(), self.tolerance.get(), self.max_iter.get()

        if func(a) * func(b) >= 0:
            messagebox.showerror("Error", "f(a) and f(b) must have opposite signs")
            return
        

        table = self.create_table(["Iter", "a", "b", "c", "f(c)"])
        c = a
        for i in range(max_iter):
            fa, fb = func(a), func(b)
            c = b - (fb * (a - b)) / (fa - fb)
            fc = func(c)
            table.insert("", "end", values=(i+1, f"{a:.6f}", f"{b:.6f}", f"{c:.6f}", f"{fc:.6e}"))
            if abs(fc) < tol: break
            if fa * fc < 0: b = c
            else: a = c

        table.pack(fill="x", pady=10)
        self.plot_graph_with_root(func, expr, c)

    def run_newton_raphson(self, func, expr):
        self.clear_output()
        x0, tol, max_iter = self.param_a.get(), self.tolerance.get(), self.max_iter.get()
        f_prime = lambdify(x, diff(expr, x), 'numpy')

        table = self.create_table(["Iter", "x", "f(x)"])
        for i in range(max_iter):
            fx, fpx = func(x0), f_prime(x0)
            if fpx == 0: break
            x1 = x0 - fx / fpx
            table.insert("", "end", values=(i+1, f"{x0:.6f}", f"{fx:.6e}"))
            if abs(func(x1)) < tol:
                x0 = x1
                break
            x0 = x1

        table.pack(fill="x", pady=10)
        self.plot_graph_with_root(func, expr, x0)

    def run_secant(self, func, expr):
        self.clear_output()
        x0, x1, tol, max_iter = self.param_a.get(), self.param_b.get(), self.tolerance.get(), self.max_iter.get()

        table = self.create_table(["Iter", "x0", "x1", "x2", "f(x2)"])
        for i in range(max_iter):
            fx0, fx1 = func(x0), func(x1)
            if fx1 - fx0 == 0: break
            x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            fx2 = func(x2)
            table.insert("", "end", values=(i+1, f"{x0:.6f}", f"{x1:.6f}", f"{x2:.6f}", f"{fx2:.6e}"))
            if abs(fx2) < tol:
                x1 = x2
                break
            x0, x1 = x1, x2

        table.pack(fill="x", pady=10)
        self.plot_graph_with_root(func, expr, x1)

    def create_table(self, columns):
        table = ttk.Treeview(self.output_frame, columns=columns, show='headings')
        for col in columns:
            table.heading(col, text=col)
        return table

    def plot_graph_with_root(self, func, expr, root):
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        x_vals = np.linspace(root - 10, root + 10, 400)
        y_vals = func(x_vals)

        ax.plot(x_vals, y_vals, label=f"$f(x) = {expr}$")
        ax.plot(root, func(root), 'ro', label=f"Root ≈ {root:.4f}")
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_title("Root Found")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.output_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_roots(self, func, expr, a, b, roots):
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        x_vals = np.linspace(a, b, 400)
        y_vals = func(x_vals)

        ax.plot(x_vals, y_vals, label=f"$f(x) = {expr}$")
        for r in roots:
            ax.plot(r, func(r), 'ro')
            ax.annotate(f"{r:.4f}", (r, func(r)), textcoords="offset points", xytext=(5, 5))
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title("Incremental Method Roots")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.output_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def clear_output(self):
        for widget in self.output_frame.winfo_children():
            widget.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = RootFinderApp(root)
    root.mainloop()