use mlrust::prelude::*;
use std::f32::consts::PI;

// ============================================================
//  Helpers
// ============================================================

/// run one SGD step on a single (input, target) pair
fn train_step<B: Backend>(
    network: &mut Network<B>,
    optimizer: &Sgd,
    loss_fn: Loss,
    x_data: Vec<f32>,
    x_shape: Vec<usize>,
    y_data: Vec<f32>,
) -> f32 {
    optimizer.zero_grad(&network.parameters());

    let x    = Tensor::<B>::from_vec(x_data,  x_shape);
    let y    = Tensor::<B>::from_vec(y_data,  vec![1]);
    let pred = network.forward(&x);
    let loss = loss_fn.compute(&pred, &y);
    let l    = loss.item();

    loss.backward();
    optimizer.step(&network.parameters());
    network.end_iter();
    l
}

// ============================================================
//  Example 1 — XOR
// ============================================================

fn run_xor() {
    println!("\n{}", "=".repeat(52));
    println!("  Example 1: XOR");
    println!("{}\n", "=".repeat(52));

    let inputs: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets: Vec<f32> = vec![0.0, 1.0, 1.0, 0.0];

    let mut network = Network::<Cpu>::new()
        .add_layer(2, 8, Activation::ReLU)
        .add_layer(8, 1, Activation::Sigmoid);

    network.print_summary();

    let optimizer = Sgd::new(0.1);
    let loss_fn   = Loss::Bce;

    for epoch in 0..3000 {
        let mut total = 0.0f32;
        for (x, &t) in inputs.iter().zip(targets.iter()) {
            total += train_step(
                &mut network, &optimizer, loss_fn,
                x.clone(), vec![2], vec![t],
            );
        }
        if epoch % 500 == 0 {
            println!("Epoch {:>4} | avg loss: {:.6}", epoch, total / 4.0);
        }
    }

    println!("\n── Inference ──────────────────────────────────");
    println!("{:<20} {:>8} {:>10} {:>8}", "Input", "Target", "Pred", "Round");
    println!("{}", "-".repeat(50));
    for (x, &t) in inputs.iter().zip(targets.iter()) {
        let inp  = Tensor::<Cpu>::from_vec(x.clone(), vec![2]);
        let pred = network.forward(&inp);
        let p    = pred.item();
        network.end_iter();
        println!(
            "{:<20} {:>8.0} {:>10.4} {:>8}",
            format!("{:?}", x), t, p,
            if p >= 0.5 { "1" } else { "0" }
        );
    }
    let correct = inputs.iter().zip(targets.iter()).filter(|(x, &t)| {
        let inp  = Tensor::<Cpu>::from_vec((*x).clone(), vec![2]);
        let pred = network.forward(&inp);
        let p    = pred.item();
        network.end_iter();
        (p >= 0.5) == (t >= 0.5)
    }).count();
    println!("\nAccuracy: {}/{} ({:.0}%)", correct, 4, correct as f32 / 4.0 * 100.0);
}

// ============================================================
//  Example 2 — Sine regression
// ============================================================

fn run_sine() {
    println!("\n{}", "=".repeat(52));
    println!("  Example 2: Sine Regression");
    println!("{}\n", "=".repeat(52));

    // ---- Generate data: 64 evenly-spaced points over [0, 2π] ----
    let n_total: usize = 128;
    let all_x: Vec<f32> = (0..n_total)
        .map(|i| i as f32 / (n_total - 1) as f32 * 2.0 * PI)
        .collect();
    let all_y: Vec<f32> = all_x.iter().map(|&x| x.sin()).collect();

    // Train on every other point (even indices), test on all
    let train_x: Vec<f32> = all_x.iter().copied().step_by(2).collect();
    let train_y: Vec<f32> = all_y.iter().copied().step_by(2).collect();
    let n_train = train_x.len();

    println!("Total points : {}", n_total);
    println!("Train points : {} (every other sample)", n_train);
    println!("Test  points : {} (all samples)\n", n_total);

    // ---- Normalise input to [0,1] so it's easier for the network ----
    // x already in [0, 2π]; divide by 2π → [0, 1]
    // y = sin(x) ∈ [-1, 1]; scale to [0, 1] for MSE stability
    let norm_x = |x: f32| x / (2.0 * PI);
    let norm_y = |y: f32| (y + 1.0) / 2.0;   // [-1,1] → [0,1]
    let denorm_y = |y: f32| y * 2.0 - 1.0;   // [0,1]  → [-1,1]

    // ---- Network -----------------------------------------------------------
    // A small MLP with Tanh hidden layers works well for smooth functions.
    let mut network = Network::<Cpu>::new()
        .add_layer(1,  16, Activation::Tanh)
        .add_layer(16, 16, Activation::Tanh)
        .add_layer(16,  1, Activation::Sigmoid);

    network.print_summary();

    let optimizer = Sgd::new(0.01);
    let loss_fn   = Loss::Mse;

    // ---- Training ----------------------------------------------------------
    let epochs = 5000;
    for epoch in 0..epochs {
        let mut total = 0.0f32;
        // Shuffle order each epoch for more stable SGD
        // (simple rotation — good enough without a random shuffle dep)
        let offset = epoch % n_train;
        let order: Vec<usize> = (offset..n_train).chain(0..offset).collect();

        for i in order {
            total += train_step(
                &mut network, &optimizer, loss_fn,
                vec![norm_x(train_x[i])], vec![1],
                vec![norm_y(train_y[i])],
            );
        }
        if epoch % 1000 == 0 {
            println!("Epoch {:>5} | avg loss: {:.6}", epoch, total / n_train as f32);
        }
    }

    // ---- Evaluation on all 64 points ------------------------------------
    println!("\n── Predictions on all {} points ──────────────────", n_total);
    println!("{:>6} {:>10} {:>10} {:>10} {:>8} {:>6}",
             "idx", "x", "y_true", "y_pred", "|error|", "split");
    println!("{}", "-".repeat(60));

    let mut train_mae = 0.0f32;
    let mut test_mae  = 0.0f32;
    let mut n_test_only = 0usize;

    for (i, (&x, &y_true)) in all_x.iter().zip(all_y.iter()).enumerate() {
        let inp  = Tensor::<Cpu>::from_vec(vec![norm_x(x)], vec![1]);
        let pred = network.forward(&inp);
        let y_pred = denorm_y(pred.item());
        network.end_iter();

        let err = (y_pred - y_true).abs();
        let is_train = i % 2 == 0;
        let split = if is_train { "train" } else { "test " };

        if is_train {
            train_mae += err;
        } else {
            test_mae      += err;
            n_test_only   += 1;
        }

        // Print every 4th row to keep output readable
        if i % 4 == 0 || i == n_total - 1 {
            println!(
                "{:>6} {:>10.4} {:>10.4} {:>10.4} {:>8.4} {:>6}",
                i, x, y_true, y_pred, err, split
            );
        }
    }

    println!("{}", "-".repeat(60));
    println!(
        "Train MAE : {:.5}  (over {} points)",
        train_mae / n_train as f32, n_train
    );
    println!(
        "Test  MAE : {:.5}  (over {} unseen points)",
        test_mae / n_test_only as f32, n_test_only
    );
    println!(
        "\nThe test MAE measures how well the network interpolates\n\
         between training points it never saw during training."
    );
}


fn main() {
    run_xor();
    run_sine();
}