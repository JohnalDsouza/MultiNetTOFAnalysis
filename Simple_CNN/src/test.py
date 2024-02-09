from tensorflow.keras.models import load_model
from Simple_CNN.src.data import test_data
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create a directory for the results if it doesn't exist
results_dir = 'Simple_CNN/CNN_results/'

os.makedirs(results_dir, exist_ok=True)
results_dir = 'Simple_CNN/CNN_results/'

# Load the best model
best_model = load_model('Simple_CNN/best_model.h5')

# Load test data
X_test, y_test = test_data(data_path="dataset/Foamboard/complex_data_test.xlsx",
                           label_path="dataset/Foamboard/raml_test_label.xlsx")

# Evaluate the model on the test data
test_results = best_model.evaluate(X_test, y_test)

# Extract MSE and MAE errors
mse_error = test_results[0]
mae_error = test_results[1]

# Specify the file path within the ANN_results folder
file_path = os.path.join(results_dir, 'test_errors.txt')

# Create or open a text file for writing in the ANN_results folder
with open(file_path, 'w') as file:
    # Write MSE and MAE errors to the file
    file.write(f'MSE error = {mse_error}\n')
    file.write(f'MAE error = {mae_error}\n')

print("MSE error = ", mse_error)
print("MAE error = ", mae_error)

# Generate predictions on test data
predictions = best_model.predict(X_test).flatten()

# Convert y_test to a NumPy array if it's a DataFrame
if isinstance(y_test, pd.DataFrame):
    y_test = y_test.to_numpy().flatten()

# Create a DataFrame with actual and predicted values
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

# Group by 'Actual' value to create separate data for each boxplot
grouped = results_df.groupby('Actual')

# Create a directory for plots if it doesn't exist
plots_dir = 'Simple_CNN/CNN_plots/'
os.makedirs(plots_dir, exist_ok=True)

# Create a boxplot for each group
plt.figure(figsize=(12, 8))
for i, (name, group) in enumerate(grouped, 1):
    plt.boxplot(group['Predicted'], positions=[i])

# Set x-axis labels to actual values (unique y_test values)
plt.xticks(range(1, len(grouped) + 1), grouped.groups.keys(), rotation=45)

plt.xlabel('Actual Thickness')
plt.ylabel('Predicted Thickness')
plt.title('Predicted vs Actual Thickness Boxplots')

# Save the plot
plt.savefig(os.path.join(plots_dir, 'predictions_boxplot.png'))
plt.close()