## Task

Implement the class `Sgd` in the file `Optimizers.py` located in the `Optimization` folder.

- The `Sgd` constructor should accept a learning rate with the data type `float`.
  
- Implement the method:
  - `calculate_update(weight_tensor, gradient_tensor)` that returns the updated weights according to the basic gradient descent update scheme.
  
You can verify your implementation using the provided testsuite by executing the command line parameter `TestOptimizers1`.

Here's an overview of Stochastic Gradient Descent (SGD):

---

## Stochastic Gradient Descent (SGD)

SGD is an optimization algorithm commonly used in machine learning and neural network training. It's a variant of the gradient descent algorithm designed to efficiently handle large datasets by processing random subsets of the data in each iteration.

### Basic Idea

- **Gradient Descent** involves updating model parameters in the direction of the negative gradient of a cost function with respect to those parameters, aiming to minimize the cost.
  
- **Stochasticity**: Unlike traditional gradient descent, SGD randomly selects a subset (or a single data point) from the entire dataset to compute the gradient at each iteration. This randomness introduces noise but helps in faster convergence and avoids getting stuck in local minima.

### Steps in SGD:

1. **Initialize Parameters**: Begin with initial parameter values for the model.
  
2. **Iterative Process**:
   - **Data Shuffling**: Randomly shuffle the dataset.
   - **Mini-batch Selection**: Select a subset (mini-batch) of data points.
   - **Compute Gradient**: Calculate the gradient of the cost function using the selected subset.
   - **Update Parameters**: Adjust the model parameters in the negative direction of the gradient with a certain learning rate.
  
3. **Repeat**: Continue this process for a defined number of iterations or until convergence criteria are met.

### Benefits of SGD:

- **Efficiency**: Particularly useful for large datasets as it processes only a subset of data in each iteration.
  
- **Fast Convergence**: Its stochastic nature can lead to faster convergence compared to batch gradient descent, especially in high-dimensional spaces.

### Parameters in SGD:

- **Learning Rate**: Controls the step size during parameter updates. A higher learning rate can cause divergence, while a lower rate might slow down convergence.
  
- **Mini-batch Size**: Determines the number of data points used in each iteration. A balance needs to be struck between computational efficiency and accurate estimation of gradients.

### Implementation Details:

- Implementing SGD involves creating a class or function that handles the iteration process, computes gradients, and updates model parameters accordingly.

### Verification:

- **Testsuite**: Provided tests such as `TestOptimizers1` can validate the correctness and effectiveness of your `Sgd` implementation by checking its behavior against expected outcomes.

### Usage:

- Utilize the implemented `Sgd` class in `Optimizers.py` by instantiating it with an appropriate learning rate and using the `calculate_update` method to update weight tensors based on provided gradients.

---
