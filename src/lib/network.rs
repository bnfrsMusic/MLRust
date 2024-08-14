use super::activations::Activation;
use super::matrix::Matrix;

use std::{
    fs::File,
    io::{Read, Write},
};

use serde::{Deserialize, Serialize};
use serde_json::{from_str, json};

pub struct Network<'a> {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    learning_rate: f64,
    activation: Activation<'a>,
}
//Derives from Serialize and Deserialize traits to allow it to be converted to and from a JSON
#[derive(Serialize, Deserialize)]
struct SaveData {
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<Vec<f64>>>,
}

impl Network<'_> {
    pub fn new<'a>(
        layers: Vec<usize>,
        learning_rate: f64,
        activation: Activation<'a>,
    ) -> Network<'a> {
        let mut weights = vec![];
        let mut biases = vec![];

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }

        Network {
            layers,
            weights,
            biases,
            data: vec![],
            learning_rate,
            activation,
        }
    }

    pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        if inputs.len() != self.layers[0] {
            panic!(
                "Invalid number of inputs: expected {}, got {}",
                self.layers[0],
                inputs.len()
            );
        }
        let mut current = Matrix::from(vec![inputs]).transpose();
        self.data = vec![current.clone()];

        for i in 0..self.layers.len() - 1 {
            current = self.weights[i]
                .multiply(&current)
                .add(&self.biases[i])
                .map(self.activation.function);
            self.data.push(current.clone());
        }
        current.data[0].to_owned()
    }

    pub fn back_propagate(&mut self, outputs: Vec<f64>, targets: Vec<f64>) {
        if targets.len() != self.layers[self.layers.len() - 1] {
            //Checks if there is a valid number of targets
            panic!("Invalid number of targets");
        }

        let mut parsed = Matrix::from(vec![outputs]);
        let mut errors = Matrix::from(vec![targets]).substract(&parsed);
        let mut gradients = parsed.map(self.activation.derivative);

        for i in (0..self.layers.len() - 1).rev() {
            // goes from last layer to first
            gradients = gradients
                .dot_multiply(&errors)
                .map(&|x| x * self.learning_rate); // multiplies errors by gradient to tell us what we did wrong

            //takes error gradiet, multiplies it by self.data (what we did before), and multiplies that by the weight. This shows us the difference between what was expected and the actual. Then we are going to be tweaking that.
            self.weights[i] = self.weights[i].add(&gradients.multiply(&self.data[i].transpose()));
            //does the similiar for bias
            self.biases[i] = self.biases[i].add(&gradients);

            //calculates error
            errors = self.weights[i].transpose().multiply(&errors);
            //reverses the activation function using its derivative
            gradients = self.data[i].map(self.activation.derivative);
        }
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u16) {
        //epochs is how many times we want to cycle through targets
        //for epchs number of times, it will loop through all the inputs, loop through all the outputs, and then feed forward the inputs, then back propogate based on the outputs
        for i in 1..epochs {
            if epochs < 100 || i % (epochs / 100) == 0 {
                println!("Epoch {} of {}", i, epochs);
            }

            for j in 0..inputs.len() {
                let outputs = self.feed_forward(inputs[j].clone());
                self.back_propagate(outputs, targets[j].clone());
            }
        }
    }

    pub fn save(&self, file: String) {
        let mut file = File::create(file).expect("Unable to write save file");

        file.write_all(json!({
            "weights": self.weights.clone().into_iter().map(|matrix| matrix.data).collect::<Vec<Vec<Vec<f64>>>>(),
            "biases": self.biases.clone().into_iter().map(|matrix| matrix.data).collect::<Vec<Vec<Vec<f64>>>>(),
        }).to_string().as_bytes()).expect("Unable to write save file");

        println!("Save File Saved Successfully");
    }

    pub fn load(&mut self, file: String) {
        let mut file = File::open(file).expect("Unable to read save file");
        let mut buffer = String::new();

        file.read_to_string(&mut buffer)
            .expect("Unable to read save file");

        let save_data: SaveData = from_str(&buffer).expect("unable to serialize save data");

        let mut weights = vec![];
        let mut biases = vec![];

        for i in 0..self.layers.len() - 1 {
            weights.push(Matrix::from(save_data.weights[i].clone()));
            biases.push(Matrix::from(save_data.biases[i].clone()));
        }

        self.weights = weights;
        self.biases = biases;

        println!("Save File Loaded Successfully");
    }
}
