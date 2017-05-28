#![feature(test)]
extern crate exponential_decay_reservoir;
extern crate test;

use exponential_decay_reservoir::ExponentialDecayReservoir;
use std::time::{Instant, Duration};
use test::Bencher;

#[bench]
fn update(b: &mut Bencher) {
    let mut reservoir = ExponentialDecayReservoir::new();

    for i in 0..1028 {
        reservoir.update(i);
    }

    b.iter(|| reservoir.update(0));

    test::black_box(reservoir.snapshot());
}

#[bench]
fn update_at(b: &mut Bencher) {
    let mut reservoir = ExponentialDecayReservoir::new();

    for i in 0..1028 {
        reservoir.update(i);
    }

    let mut now = Instant::now();
    b.iter(|| reservoir.update_at(now, 0));

    test::black_box(reservoir.snapshot());
}

#[bench]
fn snapshot(b: &mut Bencher) {
    let mut reservoir = ExponentialDecayReservoir::new();

    for i in 0..1028 {
        reservoir.update(i);
    }

    b.iter(|| reservoir.snapshot());
}

#[bench]
fn now(b: &mut Bencher) {
    b.iter(|| Instant::now());
}
