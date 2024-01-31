from tensorflow.keras.models import load_model
from data import test_data
import os

# Create a directory for the results if it doesn't exist
results_dir = 'ANN_results/'
os.makedirs(results_dir, exist_ok=True)

# Load the best model
best_model = load_model('best_model.h5')

# Load test data
X_test, y_test = test_data(data_path="Foamboard/complex_data_test.xlsx",
                           label_path="Foamboard/raml_test_label.xlsx")

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

