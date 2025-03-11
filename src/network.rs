use crate::tensor::Tensor;
use crate::activations::Activation;
use crate::loss::LossFunction;

#[derive(Clone, Debug)]
pub struct Layer {
    weights: Tensor,
    biases: Tensor,
    activation: Activation,
    // These fields store the state during forward/backward passes
    inputs: Option<Tensor>,
    z_values: Option<Tensor>,
    activations: Option<Tensor>,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        // He initialization for weights: scale by sqrt(2/n_in)
        let scale = (2.0 / input_size as f64).sqrt();
        let mut weights = Tensor::random(input_size, output_size);
        weights = &weights * scale;
        
        // Initialize biases to zeros
        let biases = Tensor::zeros(1, output_size);
        
        Layer {
            weights,
            biases,
            activation,
            inputs: None,
            z_values: None,
            activations: None,
        }
    }
    
    // Forward pass through the layer
    pub fn forward(&mut self, inputs: &Tensor) -> Tensor {
        // Store inputs for backward pass
        self.inputs = Some(inputs.clone());
        
        // Compute Z = X·W + b
        let mut z = inputs.dot(&self.weights);
        
        // Add biases - need to broadcast to each row
        let (rows, cols) = z.shape();
        for i in 0..rows {
            for j in 0..cols {
                let value = z.get_value(i, j) + self.biases.get_value(0, j);
                z.set_value(i, j, value);
            }
        }
        
        self.z_values = Some(z.clone());
        
        // Apply activation function
        let activations = self.activation.apply(&z);
        self.activations = Some(activations.clone());
        
        activations
    }
    
    // Backward pass through the layer to compute gradients
    pub fn backward(&mut self, prev_layer_error: &Tensor, learning_rate: f64) -> Tensor {
        let inputs = self.inputs.as_ref().expect("Forward pass must be called before backward pass");
        let activations = self.activations.as_ref().expect("Forward pass must be called before backward pass");
        
        // Compute the derivative of the activation function
        let activation_derivative = self.activation.derivative(activations);
        
        // Element-wise multiply the error with the activation derivative (chain rule)
        let delta = prev_layer_error.hadamard(&activation_derivative);
        
        // Compute gradients for weights: dL/dW = X^T · delta
        let weight_gradients = &inputs.transpose().dot(&delta);
        
        // Compute gradients for biases: dL/db = sum(delta, axis=0)
        let bias_gradients = &delta.sum_axis_0();
        
        // Compute error to propagate to previous layer: dL/dX = delta · W^T
        let prev_error = &delta.dot(&self.weights.transpose());
        
        // Update weights and biases using gradient descent
        self.weights = (&self.weights - &(weight_gradients * learning_rate));
        self.biases = (&self.biases - &(bias_gradients * learning_rate));
        
        prev_error.clone()
    }
}

#[derive(Clone, Debug)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(layers: Vec<Layer>) -> Self {
        Network { layers }
    }
    
    // Forward pass through the entire network
    pub fn forward(&mut self, inputs: &Tensor) -> Tensor {
        let mut current_input = inputs.clone();
        
        for layer in &mut self.layers {
            current_input = layer.forward(&current_input);
        }
        
        current_input
    }
    
    // Backward pass to update weights using backpropagation
    pub fn backward(&mut self, inputs: &Tensor, targets: &Tensor, loss_function: &LossFunction, learning_rate: f64) {
        // Forward pass to ensure we have all necessary intermediate values
        let outputs = self.forward(inputs);
        
        // Compute initial error gradient from loss function
        let mut layer_error = loss_function.derivative(&outputs, targets);
        
        // Backpropagate through layers in reverse order
        for layer in self.layers.iter_mut().rev() {
            layer_error = layer.backward(&layer_error, learning_rate);
        }
    }
}