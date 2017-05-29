#![feature(test)]
extern crate exponential_decay_histogram;
extern crate test;

use exponential_decay_histogram::ExponentialDecayHistogram;
use std::time::Instant;
use test::Bencher;

#[bench]
fn update(b: &mut Bencher) {
    let mut histogram = ExponentialDecayHistogram::new();

    for i in 0..1028 {
        histogram.update(i);
    }

    b.iter(|| histogram.update(0));

    test::black_box(histogram.snapshot());
}

#[bench]
fn update_at(b: &mut Bencher) {
    let mut histogram = ExponentialDecayHistogram::new();

    for i in 0..1028 {
        histogram.update(i);
    }

    let now = Instant::now();
    b.iter(|| histogram.update_at(now, 0));

    test::black_box(histogram.snapshot());
}

#[bench]
fn snapshot(b: &mut Bencher) {
    let mut histogram = ExponentialDecayHistogram::new();

    for i in 0..1028 {
        histogram.update(i);
    }

    b.iter(|| histogram.snapshot());
}

#[bench]
fn now(b: &mut Bencher) {
    b.iter(|| Instant::now());
}
