extern crate criterion;
extern crate exponential_decay_histogram;

use criterion::{Bencher, Criterion};
use exponential_decay_histogram::ExponentialDecayHistogram;
use std::hint;
use std::time::Instant;

fn update(b: &mut Bencher) {
    let mut histogram = ExponentialDecayHistogram::new();

    for i in 0..1028 {
        histogram.update(i);
    }

    b.iter(|| histogram.update(0));

    hint::black_box(histogram.snapshot());
}

fn update_at(b: &mut Bencher) {
    let mut histogram = ExponentialDecayHistogram::new();

    for i in 0..1028 {
        histogram.update(i);
    }

    let now = Instant::now();
    b.iter(|| histogram.update_at(now, 0));

    hint::black_box(histogram.snapshot());
}

fn snapshot(b: &mut Bencher) {
    let mut histogram = ExponentialDecayHistogram::new();

    for i in 0..1028 {
        histogram.update(i);
    }

    b.iter(|| histogram.snapshot());
}

fn now(b: &mut Bencher) {
    b.iter(Instant::now);
}

fn main() {
    Criterion::default()
        .configure_from_args()
        .bench_function("update", update)
        .bench_function("update_at", update_at)
        .bench_function("snapshot", snapshot)
        .bench_function("now", now)
        .final_summary();
}
