use ndarray::{Array1, Array2};

#[derive(Clone, Copy, Debug)]
pub enum LossFunction {
    MeanSquaredError,
    BinaryCrossEntropy,
}

impl LossFunction {
    /// loss between predictions and targets
    pub fn compute(&self, predictions: &[f64], targets: &[f64]) -> f64 {
        assert_eq!(
            predictions.len(),
            targets.len(),
            "Predictions and targets must have the same length"
        );
        
        let n = predictions.len() as f64;
        
        match self {
            LossFunction::MeanSquaredError => {
                // MSE = (1/n) * Σ(y_pred - y_true)²
                let mut total_error = 0.0;
                
                for i in 0..predictions.len() {
                    let diff = predictions[i] - targets[i];
                    total_error += diff * diff;
                }
                
                total_error / n
            }
            LossFunction::BinaryCrossEntropy => {
                // BCE = -(1/n) * Σ(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
                let mut total_error = 0.0;
                const EPSILON: f64 = 1e-15;
                
                for i in 0..predictions.len() {
                    let y_true = targets[i];
                    let y_pred = predictions[i].max(EPSILON).min(1.0 - EPSILON); // Clip to avoid log(0)
                    total_error -= y_true * y_pred.ln() + (1.0 - y_true) * (1.0 - y_pred).ln();
                }
                
                total_error / n
            }
        }
    }
    
    /// derivative of the loss function with respect to predictions
    pub fn derivative(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Array1<f64> {
        assert_eq!(
            predictions.len(),
            targets.len(),
            "Predictions and targets must have the same length"
        );
        
        let n = predictions.len() as f64;
        let mut result = Array1::zeros(predictions.len());
        
        match self {
            LossFunction::MeanSquaredError => {
                // dMSE/dy_pred = 2 * (y_pred - y_true) / n
                for i in 0..predictions.len() {
                    let diff = predictions[i] - targets[i];
                    result[i] = 2.0 * diff / n;
                }
            }
            LossFunction::BinaryCrossEntropy => {
                // dBCE/dy_pred = -(y_true/y_pred) + (1-y_true)/(1-y_pred)
                const EPSILON: f64 = 1e-15;
                
                for i in 0..predictions.len() {
                    let y_true = targets[i];
                    let y_pred = predictions[i].max(EPSILON).min(1.0 - EPSILON); // Clip to avoid division by zero
                    let derivative = -y_true / y_pred + (1.0 - y_true) / (1.0 - y_pred);
                    result[i] = derivative / n;
                }
            }
        }
        
        result
    }
    
    ///derivative for Vec<f64> (if needed)
    pub fn derivative_vec(&self, predictions: &[f64], targets: &[f64]) -> Vec<f64> {
        let pred_array = Array1::from(predictions.to_vec());
        let target_array = Array1::from(targets.to_vec());
        self.derivative(&pred_array, &target_array).to_vec()
    }

    /// loss for batched data (2D arrays)
    pub fn compute_batch(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        assert_eq!(
            predictions.shape(),
            targets.shape(),
            "Predictions and targets must have the same shape"
        );
        
        let (n_samples, n_features) = predictions.dim();
        let total_elements = (n_samples * n_features) as f64;
        
        match self {
            LossFunction::MeanSquaredError => {
                // MSE = (1/n) * Σ(y_pred - y_true)²
                let mut total_error = 0.0;
                
                for i in 0..n_samples {
                    for j in 0..n_features {
                        let diff = predictions[[i, j]] - targets[[i, j]];
                        total_error += diff * diff;
                    }
                }
                
                total_error / total_elements
            }
            LossFunction::BinaryCrossEntropy => {
                // BCE = -(1/n) * Σ(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
                let mut total_error = 0.0;
                const EPSILON: f64 = 1e-15;
                
                for i in 0..n_samples {
                    for j in 0..n_features {
                        let y_true = targets[[i, j]];
                        let y_pred = predictions[[i, j]].max(EPSILON).min(1.0 - EPSILON);
                        total_error -= y_true * y_pred.ln() + (1.0 - y_true) * (1.0 - y_pred).ln();
                    }
                }
                
                total_error / n_samples as f64
            }
        }
    }
    
    //// derivative for batched data
    pub fn derivative_batch(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        assert_eq!(
            predictions.shape(),
            targets.shape(),
            "Predictions and targets must have the same shape"
        );
        
        let (n_samples, n_features) = predictions.dim();
        let mut result = Array2::zeros((n_samples, n_features));
        
        match self {
            LossFunction::MeanSquaredError => {
                // dMSE/dy_pred = 2 * (y_pred - y_true) / n
                let total_elements = (n_samples * n_features) as f64;
                
                for i in 0..n_samples {
                    for j in 0..n_features {
                        let diff = predictions[[i, j]] - targets[[i, j]];
                        result[[i, j]] = 2.0 * diff / total_elements;
                    }
                }
            }
            LossFunction::BinaryCrossEntropy => {
                // dBCE/dy_pred = -(y_true/y_pred) + (1-y_true)/(1-y_pred)
                const EPSILON: f64 = 1e-15;
                
                for i in 0..n_samples {
                    for j in 0..n_features {
                        let y_true = targets[[i, j]];
                        let y_pred = predictions[[i, j]].max(EPSILON).min(1.0 - EPSILON);
                        let derivative = -y_true / y_pred + (1.0 - y_true) / (1.0 - y_pred);
                        result[[i, j]] = derivative / n_samples as f64;
                    }
                }
            }
        }
        
        result
    }

}