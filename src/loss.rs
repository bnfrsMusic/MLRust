use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug)]
pub enum LossFunction {
    MeanSquaredError,
    BinaryCrossEntropy,
}

impl LossFunction {
    // Compute the loss between predictions and targets
    pub fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        let (pred_rows, pred_cols) = predictions.shape();
        let (target_rows, target_cols) = targets.shape();
        
        assert_eq!(pred_rows, target_rows, "Number of samples must match");
        assert_eq!(pred_cols, target_cols, "Number of output features must match");
        
        match self {
            LossFunction::MeanSquaredError => {
                // MSE = (1/n) * Σ(y_pred - y_true)²
                let mut total_error = 0.0;
                
                for i in 0..pred_rows {
                    for j in 0..pred_cols {
                        let diff = predictions.get_value(i, j) - targets.get_value(i, j);
                        total_error += diff * diff;
                    }
                }
                
                total_error / (pred_rows as f64 * pred_cols as f64)
            },
            LossFunction::BinaryCrossEntropy => {
                // BCE = -(1/n) * Σ(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
                let mut total_error = 0.0;
                
                for i in 0..pred_rows {
                    for j in 0..pred_cols {
                        let y_true = targets.get_value(i, j);
                        let y_pred = predictions.get_value(i, j).max(1e-15).min(1.0 - 1e-15); // Clip to avoid log(0)
                        total_error -= y_true * y_pred.ln() + (1.0 - y_true) * (1.0 - y_pred).ln();
                    }
                }
                
                total_error / (pred_rows as f64)
            },
        }
    }
    
    // Compute the derivative of the loss function with respect to predictions
    pub fn derivative(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let (pred_rows, pred_cols) = predictions.shape();
        let (target_rows, target_cols) = targets.shape();
        
        assert_eq!(pred_rows, target_rows, "Number of samples must match");
        assert_eq!(pred_cols, target_cols, "Number of output features must match");
        
        let mut result = Tensor::zeros(pred_rows, pred_cols);
        
        match self {
            LossFunction::MeanSquaredError => {
                // dMSE/dy_pred = 2 * (y_pred - y_true) / n
                for i in 0..pred_rows {
                    for j in 0..pred_cols {
                        let diff = predictions.get_value(i, j) - targets.get_value(i, j);
                        // The 2.0 factor can be omitted and absorbed into the learning rate
                        let derivative = 2.0 * diff / (pred_rows as f64 * pred_cols as f64);
                        result.set_value(i, j, derivative);
                    }
                }
            },
            LossFunction::BinaryCrossEntropy => {
                // dBCE/dy_pred = - y_true/y_pred + (1-y_true)/(1-y_pred)
                for i in 0..pred_rows {
                    for j in 0..pred_cols {
                        let y_true = targets.get_value(i, j);
                        let y_pred = predictions.get_value(i, j).max(1e-15).min(1.0 - 1e-15); // Clip to avoid division by zero
                        // Fixed: using mut to allow modification
                        let mut derivative = -y_true / y_pred + (1.0 - y_true) / (1.0 - y_pred);
                        derivative /= pred_rows as f64;  // Average over batch
                        result.set_value(i, j, derivative);
                    }
                }
            },
        }
        
        result
    }
}