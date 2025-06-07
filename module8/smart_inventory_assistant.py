import numpy as np
from sklearn.naive_bayes import GaussianNB
from collections import defaultdict
from typing import Dict, List, Tuple
import datetime
import json
import os

class SmartInventoryAssistant:
    def __init__(self):
        # Initialize inventory tracking
        self.inventory: Dict[str, int] = defaultdict(int)
        self.min_stock_levels: Dict[str, int] = {}
        self.sales_history: Dict[str, List[Tuple[datetime.datetime, int]]] = defaultdict(list)
        
        # Initialize the demand prediction model
        self.demand_model = GaussianNB()
        self.is_model_trained = False
        
    def save_state(self, filename: str = "inventory_state.json") -> None:
        """Save the current state of the inventory to a JSON file."""
        state = {
            "inventory": dict(self.inventory),
            "min_stock_levels": self.min_stock_levels,
            "sales_history": {
                item: [(date.isoformat(), qty) for date, qty in sales]
                for item, sales in self.sales_history.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"\nInventory state saved to {filename}")
        
    def load_state(self, filename: str = "inventory_state.json") -> bool:
        """Load the inventory state from a JSON file."""
        if not os.path.exists(filename):
            return False
            
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
                
            self.inventory = defaultdict(int, state["inventory"])
            self.min_stock_levels = state["min_stock_levels"]
            self.sales_history = defaultdict(list)
            
            for item, sales in state["sales_history"].items():
                self.sales_history[item] = [
                    (datetime.datetime.fromisoformat(date), qty)
                    for date, qty in sales
                ]
                
            # Retrain the model with loaded data
            self.train_demand_model()
            return True
            
        except Exception as e:
            print(f"Error loading state: {str(e)}")
            return False
        
    def update_stock(self, item: str, quantity: int, operation: str = "add") -> None:
        """Update inventory levels for an item."""
        if operation == "add":
            self.inventory[item] += quantity
        elif operation == "remove":
            self.inventory[item] = max(0, self.inventory[item] - quantity)
            
    def record_sale(self, item: str, quantity: int) -> None:
        """Record a sale and update inventory."""
        self.update_stock(item, quantity, "remove")
        self.sales_history[item].append((datetime.datetime.now(), quantity))
        
    def set_min_stock_level(self, item: str, level: int) -> None:
        """Set minimum stock level for an item."""
        self.min_stock_levels[item] = level
        
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for demand prediction."""
        features = []
        labels = []
        
        for item, sales in self.sales_history.items():
            if len(sales) < 2:  # Need at least 2 data points
                continue
                
            for i in range(1, len(sales)):
                prev_date, prev_quantity = sales[i-1]
                curr_date, curr_quantity = sales[i]
                
                # Features: [month, day_of_week, previous_quantity]
                features.append([
                    prev_date.month,
                    prev_date.weekday(),
                    prev_quantity
                ])
                
                # Label: 1 for high demand (above average), 0 for low demand
                avg_quantity = np.mean([q for _, q in sales])
                labels.append(1 if curr_quantity > avg_quantity else 0)
                
        return np.array(features), np.array(labels)
        
    def train_demand_model(self) -> None:
        """Train the demand prediction model."""
        if len(self.sales_history) < 2:
            print("Not enough sales data to train the model")
            return
            
        X, y = self.prepare_training_data()
        if len(X) > 0:
            self.demand_model.fit(X, y)
            self.is_model_trained = True
            print("Demand prediction model trained successfully")
            
    def predict_demand(self, item: str) -> str:
        """Predict if an item will have high or low demand."""
        if not self.is_model_trained or item not in self.sales_history:
            return "insufficient_data"
            
        sales = self.sales_history[item]
        if len(sales) < 2:
            return "insufficient_data"
            
        last_sale_date, last_quantity = sales[-1]
        features = np.array([[
            last_sale_date.month,
            last_sale_date.weekday(),
            last_quantity
        ]])
        
        prediction = self.demand_model.predict(features)[0]
        return "high" if prediction == 1 else "low"
        
    def get_restock_suggestions(self) -> List[Dict]:
        """Get suggestions for items that need restocking."""
        suggestions = []
        
        for item, current_stock in self.inventory.items():
            if item not in self.min_stock_levels:
                continue
                
            min_level = self.min_stock_levels[item]
            if current_stock < min_level:
                demand = self.predict_demand(item)
                suggested_quantity = min_level - current_stock
                
                # Adjust suggested quantity based on predicted demand
                if demand == "high":
                    suggested_quantity = int(suggested_quantity * 1.5)
                    
                suggestions.append({
                    "item": item,
                    "current_stock": current_stock,
                    "min_level": min_level,
                    "suggested_quantity": suggested_quantity,
                    "predicted_demand": demand
                })
                
        return suggestions

def handle_user_input(assistant: SmartInventoryAssistant) -> None:
    """Handle interactive user input for the inventory assistant."""
    print("\nSmart Inventory Assistant - Interactive Mode")
    print("Available commands:")
    print("  restock <item> <quantity>  - Add stock to inventory")
    print("  sell <item> <quantity>     - Record a sale")
    print("  setmin <item> <level>      - Set minimum stock level")
    print("  predict <item>             - Show demand prediction")
    print("  suggest                    - Show restock suggestions")
    print("  inventory                  - Show current inventory")
    print("  save                       - Save current state")
    print("  help                       - Show this help message")
    print("  exit                       - Save and exit the program")
    
    while True:
        try:
            command = input("\nEnter command: ").strip().lower()
            
            if command == "exit":
                assistant.save_state()
                print("Goodbye!")
                break
            elif command == "save":
                assistant.save_state()
            elif command == "help":
                print("\nAvailable commands:")
                print("  restock <item> <quantity>  - Add stock to inventory")
                print("  sell <item> <quantity>     - Record a sale")
                print("  setmin <item> <level>      - Set minimum stock level")
                print("  predict <item>             - Show demand prediction")
                print("  suggest                    - Show restock suggestions")
                print("  inventory                  - Show current inventory")
                print("  save                       - Save current state")
                print("  help                       - Show this help message")
                print("  exit                       - Save and exit the program")
            
            elif command == "inventory":
                print("\nCurrent Inventory:")
                for item, qty in assistant.inventory.items():
                    print(f"  {item}: {qty}")
            
            elif command == "suggest":
                suggestions = assistant.get_restock_suggestions()
                print("\nRestock Suggestions:")
                if not suggestions:
                    print("  No items need restocking.")
                for suggestion in suggestions:
                    print(f"\nItem: {suggestion['item']}")
                    print(f"  Current Stock: {suggestion['current_stock']}")
                    print(f"  Minimum Level: {suggestion['min_level']}")
                    print(f"  Suggested Quantity: {suggestion['suggested_quantity']}")
                    print(f"  Predicted Demand: {suggestion['predicted_demand']}")
            
            else:
                parts = command.split()
                if len(parts) < 2:
                    print("Invalid command. Type 'help' for available commands.")
                    continue
                
                action, item = parts[0], parts[1]
                
                if action == "restock":
                    if len(parts) != 3:
                        print("Usage: restock <item> <quantity>")
                        continue
                    try:
                        quantity = int(parts[2])
                        assistant.update_stock(item, quantity)
                        print(f"Added {quantity} to {item}. New stock: {assistant.inventory[item]}")
                    except ValueError:
                        print("Quantity must be a number.")
                
                elif action == "sell":
                    if len(parts) != 3:
                        print("Usage: sell <item> <quantity>")
                        continue
                    try:
                        quantity = int(parts[2])
                        if item not in assistant.inventory:
                            print(f"Item '{item}' not found in inventory.")
                            continue
                        if assistant.inventory[item] < quantity:
                            print(f"Not enough stock. Current stock: {assistant.inventory[item]}")
                            continue
                        assistant.record_sale(item, quantity)
                        print(f"Sold {quantity} of {item}. New stock: {assistant.inventory[item]}")
                    except ValueError:
                        print("Quantity must be a number.")
                
                elif action == "setmin":
                    if len(parts) != 3:
                        print("Usage: setmin <item> <level>")
                        continue
                    try:
                        level = int(parts[2])
                        assistant.set_min_stock_level(item, level)
                        print(f"Minimum stock for {item} set to {level}")
                    except ValueError:
                        print("Level must be a number.")
                
                elif action == "predict":
                    if item not in assistant.inventory:
                        print(f"Item '{item}' not found in inventory.")
                        continue
                    demand = assistant.predict_demand(item)
                    print(f"\nDemand prediction for {item}: {demand}")
                
                else:
                    print("Invalid command. Type 'help' for available commands.")
        
        except Exception as e:
            print(f"An error occurred: {str(e)}")

def main():
    # Initialize the assistant
    assistant = SmartInventoryAssistant()
    
    # Try to load previous state
    if assistant.load_state():
        print("Loaded previous inventory state.")
    else:
        print("No previous state found. Starting with sample data.")
        # Set minimum stock levels for multiple items
        assistant.set_min_stock_level("widget", 10)
        assistant.set_min_stock_level("gadget", 5)
        assistant.set_min_stock_level("doodad", 8)
        assistant.set_min_stock_level("thingamajig", 12)
        
        # Add initial stock
        assistant.update_stock("widget", 7)
        assistant.update_stock("gadget", 3)
        assistant.update_stock("doodad", 10)
        assistant.update_stock("thingamajig", 15)
        
        # Record sales for different days and quantities
        now = datetime.datetime.now()
        sales_data = [
            ("widget", [(now - datetime.timedelta(days=10), 2), (now - datetime.timedelta(days=5), 4), (now, 3)]),
            ("gadget", [(now - datetime.timedelta(days=8), 1), (now - datetime.timedelta(days=3), 2), (now, 1)]),
            ("doodad", [(now - datetime.timedelta(days=7), 5), (now - datetime.timedelta(days=2), 2), (now, 3)]),
            ("thingamajig", [(now - datetime.timedelta(days=12), 6), (now - datetime.timedelta(days=6), 5), (now, 4)])
        ]
        for item, sales in sales_data:
            for sale_date, qty in sales:
                assistant.sales_history[item].append((sale_date, qty))
        
        # Train the model
        assistant.train_demand_model()
    
    # Start interactive mode
    handle_user_input(assistant)

if __name__ == "__main__":
    main()
