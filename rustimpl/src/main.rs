/////////////////////////////////////////////////////////////////////////////////
/// 
/// WebGPU code replicated in rust (for benchmarking)
/// 
////////////////////////////////////////////////////////////////////////////////

use rand::distributions::{Distribution, Uniform};
use std::time::Instant;

const GPU_INNER_LOOP_ITERATIONS : i32 = 10000;

fn pcg_rng(input: u32) -> u32 {
    let state = input * 747796405u32 + 2891336453u32;
    let word = ((state >> ((state >> 28u32) + 4u32)) ^ state) * 277803737u32;
    return (word >> 22u32) ^ word;
}

fn calc(randval01 : f32) -> f32 {
    return f32::sqrt(1.0f32-(randval01*randval01));
}

fn shader_main(random_seed : i32) -> f32 {
    // estimate pi over 10k iterations
    let mut rand : u32 = random_seed as u32;
    let mut pi_ests_sum : f32 = 0f32;
    let mut pi_ests_count : f32 = 0f32;
    for _ in 1..(GPU_INNER_LOOP_ITERATIONS+1) {
        rand = pcg_rng(rand);
        let randval01 = (rand as f32) * (1.0 / 4294967295.0);
        pi_ests_sum += calc(randval01);
        pi_ests_count += 1f32;
    }
    let pi_est : f32 = 4.0 * (pi_ests_sum / pi_ests_count);
    return pi_est;
}

fn main() {
    let t = Instant::now();

    const MAX_I32 : i32 = 2147483647i32;
    let mut rng = rand::thread_rng();
    let uniform_dist = Uniform::from(1i32..MAX_I32);

    const NUM_SEEDS : usize = 100_000usize;
    let mut pi_est_sum : f64 = 0f64;
    let mut pi_est_count : f64 = 0f64;
    for _ in 1..(NUM_SEEDS+1) {
        let rand_seed : i32 = uniform_dist.sample(&mut rng);
        let pi_est : f32 = shader_main(rand_seed);
        pi_est_sum += pi_est as f64;
        pi_est_count += 1.0f64;
    }
    let pi_est_mean : f64 = pi_est_sum / pi_est_count;
    println!("pi ~= {:.6} ({} iterations in {:?})", pi_est_mean, 
        pi_est_count*(GPU_INNER_LOOP_ITERATIONS as f64),
        t.elapsed());
}