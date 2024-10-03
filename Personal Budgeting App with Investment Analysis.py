import sqlite3
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
import hashlib

class BudgetingApp:
    def __init__(self):
        self.db = sqlite3.connect("budgeting_app.db")
        self.cursor = self.db.cursor()
        self.root = tk.Tk()
        self.root.title("Budgeting App")
        self.login()

    def login(self):
        self.login_frame = tk.Frame(self.root)
        self.login_frame.pack()
        tk.Label(self.login_frame, text="Username:").pack()
        self.username_entry = tk.Entry(self.login_frame)
        self.username_entry.pack()
        tk.Label(self.login_frame, text="Password:").pack()
        self.password_entry = tk.Entry(self.login_frame, show="*")
        self.password_entry.pack()
        tk.Button(self.login_frame, text="Login", command=self.check_login).pack()
        tk.Button(self.login_frame, text="Register", command=self.register).pack()

    def check_login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        self.cursor.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")
        self.cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_password))
        user_data = self.cursor.fetchone()
        if user_data:
            self.main_screen()
        else:
            messagebox.showerror("Invalid Credentials", "Invalid username or password")

    def register(self):
        self.register_frame = tk.Frame(self.root)
        self.register_frame.pack()
        tk.Label(self.register_frame, text="Username:").pack()
        self.username_entry = tk.Entry(self.register_frame)
        self.username_entry.pack()
        tk.Label(self.register_frame, text="Password:").pack()
        self.password_entry = tk.Entry(self.register_frame, show="*")
        self.password_entry.pack()
        tk.Button(self.register_frame, text="Register", command=self.check_register).pack()

    def check_register(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        self.cursor.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")
        self.cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        existing_user = self.cursor.fetchone()
        if existing_user:
            messagebox.showerror("Username Already Exists", "Username already exists")
        else:
            self.cursor.execute("INSERT INTO users VALUES (?, ?)", (username, hashed_password))
            self.db.commit()
            messagebox.showinfo("Registration Successful", "Registration successful")
            self.login()

    def main_screen(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack()
        tk.Button(self.main_frame, text="Add Transaction", command=self.add_transaction).pack()
        tk.Button(self.main_frame, text="Add Budget", command=self.add_budget).pack()
        tk.Button(self.main_frame, text="Add Investment", command=self.add_investment).pack()
        tk.Button(self.main_frame, text="View Budget", command=self.view_budget).pack()
        tk.Button(self.main_frame, text="View Investment", command=self.view_investment).pack()

    def add_transaction(self):
        self.transaction_frame = tk.Frame(self.root)
        self.transaction_frame.pack()
        tk.Label(self.transaction_frame, text="Date:").pack()
        self.date_entry = tk.Entry(self.transaction_frame)
        self.date_entry.pack()
        tk.Label(self.transaction_frame, text="Amount:").pack()
        self.amount_entry = tk.Entry(self.transaction_frame)
        self.amount_entry.pack()
        tk.Label(self.transaction_frame, text="Category:").pack()
        self.category_entry = tk.Entry(self.transaction_frame)
        self.category_entry.pack()
        tk.Button(self.transaction_frame, text="Add Transaction", command=self.add_transaction_to_db).pack()

    def add_transaction_to_db(self):
        date = self.date_entry.get()
        amount = float(self.amount_entry.get())
        category = self.category_entry.get()
        self.cursor.execute("CREATE TABLE IF NOT EXISTS transactions (date DATE, amount REAL, category TEXT)")
        self.cursor.execute("INSERT INTO transactions VALUES (?, ?, ?)", (date, amount, category))
        self.db.commit()
        self.main_screen()

    def add_budget(self):
        self.budget_frame = tk.Frame(self.root)
        self.budget_frame.pack()
        tk.Label(self.budget_frame, text="Income:").pack()
        self.income_entry = tk.Entry(self.budget_frame)
        self.income_entry.pack()
        tk.Label(self.budget_frame, text="Expenses:").pack()
        self.expenses_entry = tk.Entry(self.budget_frame)
        self.expenses_entry.pack()
        tk.Button(self.budget_frame, text="Add Budget", command=self.add_budget_to_db).pack()

    def add_budget_to_db(self):
        income = float(self.income_entry.get())
        expenses = float(self.expenses_entry.get())
        self.cursor.execute("CREATE TABLE IF NOT EXISTS budgets (income REAL, expenses REAL)")
        self.cursor.execute("INSERT INTO budgets VALUES (?, ?)", (income, expenses))
        self.db.commit()
        self.main_screen()

    def add_investment(self):
        self.investment_frame = tk.Frame(self.root)
        self.investment_frame.pack()
        tk.Label(self.investment_frame, text="Name:").pack()
        self.name_entry = tk.Entry(self.investment_frame)
        self.name_entry.pack()
        tk.Label(self.investment_frame, text="Amount:").pack()
        self.amount_entry = tk.Entry(self.investment_frame)
        self.amount_entry.pack()
        tk.Label(self.investment_frame, text="Returns:").pack()
        self.returns_entry = tk.Entry(self.investment_frame)
        self.returns_entry.pack()
        tk.Button(self.investment_frame, text="Add Investment", command=self.add_investment_to_db).pack()

    def add_investment_to_db(self):
        name = self.name_entry.get()
        amount = float(self.amount_entry.get())
        returns = float(self.returns_entry.get())
        self.cursor.execute("CREATE TABLE IF NOT EXISTS investments (name TEXT, amount REAL, returns REAL)")
        self.cursor.execute("INSERT INTO investments VALUES (?, ?, ?)", (name, amount, returns))
        self.db.commit()
        self.main_screen()

    def view_budget(self):
        self.budget_frame = tk.Frame(self.root)
        self.budget_frame.pack()
        self.cursor.execute("SELECT * FROM budgets")
        budget_data = self.cursor.fetchone()
        if budget_data:
            income = budget_data[0]
            expenses = budget_data[1]
            tk.Label(self.budget_frame, text=f"Income: {income}").pack()
            tk.Label(self.budget_frame, text=f"Expenses: {expenses}").pack()
        else:
            tk.Label(self.budget_frame, text="No budget data available").pack()

    def view_investment(self):
        self.investment_frame = tk.Frame(self.root)
        self.investment_frame.pack()
        self.cursor.execute("SELECT * FROM investments")
        investment_data = self.cursor.fetchone()
        if investment_data:
            name = investment_data[0]
            amount = investment_data[1]
            returns = investment_data[2]
            tk.Label(self.investment_frame, text=f"Name: {name}").pack()
            tk.Label(self.investment_frame, text=f"Amount: {amount}").pack()
            tk.Label(self.investment_frame, text=f"Returns: {returns}").pack()
        else:
            tk.Label(self.investment_frame, text="No investment data available").pack()

if __name__ == "__main__":
    app = BudgetingApp()
    app.root.mainloop()