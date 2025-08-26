# Neural Network Playground

A simple, interactive web application built with Streamlit to visualize and experiment with a small Multi-Layer Perceptron (MLP) neural network for binary classification. This project allows users to generate synthetic datasets, customize the neural network architecture, and observe how the decision boundary and training loss evolve during training.

## Features

- **Interactive Controls**: Adjust dataset type (moons, circles, blobs), sample size, noise level, and random seed via a user-friendly sidebar.
- **Customizable Neural Network**: Configure the number of hidden layers, neurons per layer, hidden layer activation function (ReLU, tanh, sigmoid), learning rate, and training epochs.
- **Visualizations**:
  - Preview the raw dataset as a scatter plot.
  - Plot the training loss curve over epochs.
  - Visualize the decision boundary with a probability heatmap and overlaid data points.
- **Metrics**: Displays training accuracy and the total number of model parameters.
- **From Scratch Implementation**: The MLP is implemented using NumPy, including forward/backward propagation and gradient descent, without relying on deep learning frameworks.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd neural-network-playground
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Install the required packages using pip:
   ```bash
   pip install streamlit numpy matplotlib scikit-learn
   ```

3. **Run the Application**:
   Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   Replace `app.py` with the name of your Python script containing the provided code.

## Usage

1. **Open the App**:
   After running the Streamlit command, the app will open in your default web browser.

2. **Configure the Dataset**:
   - Select a dataset type (moons, circles, or blobs).
   - Adjust the number of samples, noise level, and random seed.
   - Click **Regenerate Data** to create a new dataset.

3. **Customize the Model**:
   - Choose the hidden layer activation function (ReLU, tanh, or sigmoid).
   - Set the number of hidden layers (1–4) and neurons per layer (2–64).
   - Select a learning rate and number of training epochs.

4. **Train and Visualize**:
   - Click **Train / Retrain** to train the model.
   - View the loss curve, decision boundary, training accuracy, and model size in the main panel.

5. **Tips**:
   - For noisy or complex datasets, try increasing the number of hidden layers or neurons.
   - If training is unstable, reduce the learning rate or increase the number of epochs.
   - Experiment with different activation functions to see their impact on the decision boundary.

## Project Structure

- `app.py`: The main Python script containing the Streamlit app and MLP implementation.
- `README.md`: This file, providing an overview and instructions.

## Requirements

- Python 3.8 or higher
- Streamlit
- NumPy
- Matplotlib
- Scikit-learn

## Notes

- The MLP is implemented from scratch using NumPy for educational purposes, with He/Xavier initialization, binary cross-entropy loss, and gradient descent.
- The app is designed for binary classification tasks on 2D synthetic datasets, making it easy to visualize decision boundaries.
- The decision boundary is plotted as a probability heatmap, with data points overlaid for clarity.

## Limitations

- The app is designed for small-scale, 2D datasets and may not scale well to high-dimensional or large datasets.
- Training is performed on the CPU using NumPy, which may be slower for large datasets or complex models.
- No model persistence is implemented; retraining is required on each run or data regeneration.

## Future Improvements

- Add support for saving and loading trained models.
- Include additional dataset types or real-world datasets.
- Implement mini-batch gradient descent for improved training efficiency.
- Add regularization options (e.g., L2 regularization, dropout) to prevent overfitting.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.