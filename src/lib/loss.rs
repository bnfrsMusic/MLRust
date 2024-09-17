use super::{activations::Activation, tensor::Tensor};

pub struct LossFunction {
    pub name: &'static str,
    pub function: fn(&Tensor, &Tensor) -> f64,
    pub derivative: fn(&Tensor, &Tensor, &Activation<'static>) -> Tensor,
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
        diff.data.iter().sum::<f64>() / (predicted.data.len() as f64)
    },
    derivative: |actual: &Tensor, predicted: &Tensor, act: &Activation<'static>| {
        let mut der = actual.clone().map(act.derivative);
        let mut cost = actual.clone();

        //makes sure the actual and predicted shape are the same (just in case I screwed up somewhere)
        assert_eq!(actual.shape, predicted.shape);

        cost.substract(predicted);
        der.multiply(&cost);
        der.map(&|x| x * 2.0);

        der
    },
};
