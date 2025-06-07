# Smart Inventory Assistant

An AI-powered inventory management system for small warehouses that uses machine learning to predict demand and suggest optimal restocking actions.

## Features

- Real-time inventory tracking
- Demand prediction using Naive Bayes classification
- Smart restocking suggestions based on current stock levels and predicted demand
- Simple text-based interface for inventory management

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the program:
```bash
python smart_inventory_assistant.py
```

## Usage

The program provides the following functionality:

1. **Inventory Management**
   - Track current stock levels
   - Update inventory (add/remove items)
   - Set minimum stock levels

2. **Demand Prediction**
   - Uses historical sales data to predict future demand
   - Classifies items as having high or low demand

3. **Restocking Suggestions**
   - Automatically suggests restocking quantities
   - Takes into account minimum stock levels and predicted demand

## Example

```python
# Create an instance of the assistant
assistant = SmartInventoryAssistant()

# Set minimum stock levels
assistant.set_min_stock_level("widget", 10)

# Add stock
assistant.update_stock("widget", 15)

# Record sales
assistant.record_sale("widget", 3)

# Get restocking suggestions
suggestions = assistant.get_restock_suggestions()
```

## Requirements

- Python 3.7+
- numpy
- scikit-learn