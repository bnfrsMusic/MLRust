use super::tensor::Tensor;

pub struct LossFunction {
    pub name: &'static str,
    pub function: fn(&Tensor, &Tensor) -> f64,
    //pub derivative: fn(&Tensor, &Tensor) -> Tensor,
}

pub const MSE: LossFunction = LossFunction {
    name: "Mean Squared Error",
    function: |predicted: &Tensor, actual: &Tensor| {
        //Function

        //MSE = 1/n Sum[i=1] (actual - predicted)^2
        let mut diff = predicted.clone();

        //(actual - predicted)^2
        diff.substract(actual); //calculates the difference
        diff.dot(&diff); //Squares the difference

        //Sum[i=1] / n
        diff.data.iter().sum::<f64>() / (predicted.data.len() as f64) //
    },
};
