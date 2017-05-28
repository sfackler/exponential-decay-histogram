extern crate rand;
extern crate ordered_float;

use ordered_float::NotNaN;
use std::collections::BTreeMap;
use std::time::{Instant, Duration};
use rand::{Rng, Open01, XorShiftRng};

const DEFAULT_SIZE: usize = 1028;
const DEFAULT_ALPHA: f64 = 0.015;
const RESCALE_THRESHOLD_SECS: u64 = 60 * 60;

struct WeightedSample {
    value: i64,
    weight: f64,
}

pub struct ExponentialDecayReservoir {
    values: BTreeMap<NotNaN<f64>, WeightedSample>,
    alpha: f64,
    size: usize,
    start_time: Instant,
    next_scale_time: Instant,
    rng: XorShiftRng,
}

impl Default for ExponentialDecayReservoir {
    fn default() -> ExponentialDecayReservoir {
        ExponentialDecayReservoir::new()
    }
}

impl ExponentialDecayReservoir {
    /// Returns a new reservoir with a default configuration.
    ///
    /// The default size is 1028, which offers a 99.9% confidence level with a
    /// 5% margin of error. The default alpha is 0.015, which heavily biases
    /// towards the last 5 minutes of values.
    pub fn new() -> ExponentialDecayReservoir {
        ExponentialDecayReservoir::from_size_and_alpha(DEFAULT_SIZE, DEFAULT_ALPHA)
    }

    /// Returns a new reservoir configured with the specified size and alpha.
    ///
    /// `size` specifies the number of elements which will be stored in the
    /// reservoir. A larger size will provide more accurate measurements, but
    /// with a higher memory overhead.
    ///
    /// `alpha` specifies the exponential decay factor. A larger factor biases
    /// the reservoir towards newer values.
    pub fn from_size_and_alpha(size: usize, alpha: f64) -> ExponentialDecayReservoir {
        let now = Instant::now();

        ExponentialDecayReservoir {
            values: BTreeMap::new(),
            alpha: alpha,
            size: size,
            start_time: now,
            next_scale_time: now + Duration::from_secs(RESCALE_THRESHOLD_SECS),
            rng: rand::thread_rng().gen(),
        }
    }

    /// Inserts a value into the reservoir at the current time.
    pub fn update(&mut self, value: i64) {
        self.update_at(Instant::now(), value);
    }

    /// Inserts a value into the reservoir at the specified time.
    ///
    /// # Panics
    ///
    /// May panic if values are inserted at non-monotonically increasing times.
    pub fn update_at(&mut self, time: Instant, value: i64) {
        self.rescale_if_needed(time);

        let item_weight = self.weight(time);
        let sample = WeightedSample {
            value,
            weight: item_weight,
        };
        // Open01 since we don't want to divide by 0
        let priority = item_weight / self.rng.gen::<Open01<f64>>().0;
        let priority = NotNaN::from(priority);

        if self.values.len() < self.size {
            self.values.insert(priority, sample);
        } else {
            let first = *self.values.keys().next().unwrap();
            if first < priority && self.values.insert(priority, sample).is_none() {
                self.values.remove(&first).unwrap();
            }
        }
    }

    /// Takes a snapshot of the current state of the reservoir.
    pub fn snapshot(&self) -> Snapshot {
        let mut entries = self.values
            .values()
            .map(|s| {
                     SnapshotEntry {
                         value: s.value,
                         norm_weight: s.weight,
                         quantile: NotNaN::from(0.),
                     }
                 })
            .collect::<Vec<_>>();

        entries.sort_by_key(|e| e.value);

        let sum_weight = entries.iter().map(|e| e.norm_weight).sum::<f64>();
        for entry in &mut entries {
            entry.norm_weight /= sum_weight;
        }

        entries
            .iter_mut()
            .fold(NotNaN::from(0.), |acc, e| {
                e.quantile = acc;
                acc + e.norm_weight
            });

        Snapshot(entries)
    }

    fn weight(&self, time: Instant) -> f64 {
        (self.alpha * (time - self.start_time).as_secs() as f64).exp()
    }

    fn rescale_if_needed(&mut self, now: Instant) {
        if now >= self.next_scale_time {
            self.rescale(now);
        }
    }

    fn rescale(&mut self, now: Instant) {
        self.next_scale_time = now + Duration::from_secs(RESCALE_THRESHOLD_SECS);
        let old_start_time = self.start_time;
        self.start_time = now;
        let scaling_factor = (-self.alpha * (now - old_start_time).as_secs() as f64).exp();

        self.values = self.values
            .iter()
            .map(|(&k, v)| {
                     (k * scaling_factor,
                      WeightedSample {
                          value: v.value,
                          weight: v.weight * scaling_factor,
                      })
                 })
            .collect();
    }
}

struct SnapshotEntry {
    value: i64,
    norm_weight: f64,
    quantile: NotNaN<f64>,
}

pub struct Snapshot(Vec<SnapshotEntry>);

impl Snapshot {
    /// Returns the value at a specified quantile in the snapshot, or 0 if it is
    /// empty.
    ///
    /// For example, `snapshot.quantile(0.5)` returns the median value of the
    /// snapshot.
    ///
    /// # Panics
    ///
    /// Panics if `quantile` is not between 0 and 1 (inclusive).
    pub fn value(&self, quantile: f64) -> i64 {
        assert!(quantile >= 0. && quantile <= 1.);

        if self.0.is_empty() {
            return 0;
        }

        let quantile = NotNaN::from(quantile);
        let idx = match self.0.binary_search_by(|e| e.quantile.cmp(&quantile)) {
            Ok(idx) => idx,
            Err(idx) if idx >= self.0.len() => self.0.len() - 1,
            Err(idx) => idx,
        };

        self.0[idx].value
    }

    /// Returns the largest value in the snapshot, or 0 if it is empty.
    pub fn max(&self) -> i64 {
        self.0.last().map_or(0, |e| e.value)
    }

    /// Returns the smallest value in the snapshot, or 0 if it is empty.
    pub fn min(&self) -> i64 {
        self.0.first().map_or(0, |e| e.value)
    }

    /// Returns the mean of the values in the snapshot, or 0 if it is empty.
    pub fn mean(&self) -> f64 {
        self.0
            .iter()
            .map(|e| e.value as f64 * e.norm_weight)
            .sum::<f64>()
    }

    /// Returns the standard deviation of the values in the snapshot, or 0 if it
    /// is empty.
    pub fn stddev(&self) -> f64 {
        if self.0.len() <= 1 {
            return 0.;
        }

        let mean = self.mean();
        let variance = self.0
            .iter()
            .map(|e| {
                     let diff = e.value as f64 - mean;
                     e.norm_weight * diff * diff
                 })
            .sum::<f64>();

        variance.sqrt()
    }
}

#[cfg(test)]
mod test {
    use std::ops::Range;

    use super::*;

    #[test]
    fn a_reservoir_of_100_out_of_1000_elements() {
        let mut reservoir = ExponentialDecayReservoir::from_size_and_alpha(100, 0.99);
        for i in 0..1000 {
            reservoir.update(i);
        }

        assert_eq!(reservoir.values.len(), 100);

        let snapshot = reservoir.snapshot();

        assert_eq!(snapshot.0.len(), 100);

        assert_all_values_between(snapshot, 0..1000);
    }

    #[test]
    fn a_reservoir_of_100_out_of_10_elements() {
        let mut reservoir = ExponentialDecayReservoir::from_size_and_alpha(100, 0.99);
        for i in 0..10 {
            reservoir.update(i);
        }

        let snapshot = reservoir.snapshot();

        assert_eq!(snapshot.0.len(), 10);

        assert_all_values_between(snapshot, 0..10);
    }

    #[test]
    fn a_heavily_biased_reservoir_of_100_out_of_1000_elements() {
        let mut reservoir = ExponentialDecayReservoir::from_size_and_alpha(1000, 0.01);
        for i in 0..100 {
            reservoir.update(i);
        }

        assert_eq!(reservoir.values.len(), 100);

        let snapshot = reservoir.snapshot();

        assert_eq!(snapshot.0.len(), 100);

        assert_all_values_between(snapshot, 0..100);
    }

    #[test]
    fn long_periods_of_inactivity_should_not_corrupt_sampling_state() {
        let mut reservoir = ExponentialDecayReservoir::from_size_and_alpha(10, 0.015);
        let mut now = reservoir.start_time;

        // add 1000 values at a rate of 10 values/second
        let delta = Duration::from_millis(100);
        for i in 0..1000 {
            now += delta;
            reservoir.update_at(now, 1000 + i);
        }

        let snapshot = reservoir.snapshot();
        assert_eq!(snapshot.0.len(), 10);
        assert_all_values_between(snapshot, 1000..2000);

        // wait for 15 hours and add another value.
        // this should trigger a rescale. Note that the number of samples will
        // be reduced because of the very small scaling factor that will make
        // all existing priorities equal to zero after rescale.
        now += Duration::from_secs(15 * 60 * 60);
        reservoir.update_at(now, 2000);

        let snapshot = reservoir.snapshot();
        assert_eq!(snapshot.0.len(), 2);
        assert_all_values_between(snapshot, 1000..3000);

        // add 1000 values at a rate of 10 values/second
        for i in 0..1000 {
            now += delta;
            reservoir.update_at(now, 3000 + i);
        }
        let snapshot = reservoir.snapshot();
        assert_eq!(snapshot.0.len(), 10);
        assert_all_values_between(snapshot, 3000..4000);
    }

    #[test]
    fn spot_lift() {
        let mut reservoir = ExponentialDecayReservoir::from_size_and_alpha(1000, 0.015);
        let mut now = reservoir.start_time;

        let values_per_minute = 10;
        let values_interval = Duration::from_secs(60) / values_per_minute;
        // mode 1: steady regime for 120 minutes
        for _ in 0..120 * values_per_minute {
            reservoir.update_at(now, 177);
            now += values_interval;
        }

        // switching to mode 2: 10 minutes with the same rate, but larger value
        for _ in 0..10 * values_per_minute {
            reservoir.update_at(now, 9999);
            now += values_interval;
        }

        // expect that the quantiles should be about mode 2 after 10 minutes
        assert_eq!(reservoir.snapshot().value(0.5), 9999);
    }

    #[test]
    fn spot_fall() {
        let mut reservoir = ExponentialDecayReservoir::from_size_and_alpha(1000, 0.015);
        let mut now = reservoir.start_time;

        let values_per_minute = 10;
        let values_interval = Duration::from_secs(60) / values_per_minute;
        // mode 1: steady regime for 120 minutes
        for _ in 0..120 * values_per_minute {
            reservoir.update_at(now, 9998);
            now += values_interval;
        }

        // switching to mode 2: 10 minutes with the same rate, but smaller value
        for _ in 0..10 * values_per_minute {
            reservoir.update_at(now, 178);
            now += values_interval;
        }

        // expect that the quantiles should be about mode 2 after 10 minutes
        assert_eq!(reservoir.snapshot().value(0.5), 178);
    }

    #[test]
    fn quantiles_should_be_based_on_weights() {
        let mut reservoir = ExponentialDecayReservoir::from_size_and_alpha(1000, 0.015);
        let mut now = reservoir.start_time;

        for _ in 0..40 {
            reservoir.update_at(now, 177);
        }

        now += Duration::from_secs(120);

        for _ in 0..10 {
            reservoir.update_at(now, 9999);
        }

        let snapshot = reservoir.snapshot();
        assert_eq!(snapshot.0.len(), 50);

        // the first added 40 items (177) have weights 1
        // the next added 10 items (9999) have weights ~6
        // so, it's 40 vs 60 distribution, not 40 vs 10
        assert_eq!(snapshot.value(0.5), 9999);
        assert_eq!(snapshot.value(0.75), 9999);
    }

    fn assert_all_values_between(snapshot: Snapshot, range: Range<i64>) {
        for entry in &snapshot.0 {
            assert!(entry.value >= range.start && entry.value < range.end,
                    "snapshot value {} was not in {:?}",
                    entry.value,
                    range);
        }
    }
}
