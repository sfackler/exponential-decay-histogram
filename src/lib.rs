extern crate rand;
extern crate ordered_float;

use ordered_float::NotNaN;
use std::collections::BTreeMap;
use std::time::{Instant, Duration};
use rand::{Rng, Open01};

const DEFAULT_SIZE: usize = 1028;
const DEFAULT_ALPHA: f64 = 0.015;
const RESCALE_THRESHOLD_SECS: u64 = 60 * 60;

struct WeightedSample {
    value: i64,
    weight: f64,
}

pub struct ForwardDecayReservoir {
    values: BTreeMap<NotNaN<f64>, WeightedSample>,
    alpha: f64,
    size: usize,
    start_time: Instant,
    next_scale_time: Instant,
}

impl ForwardDecayReservoir {
    pub fn new() -> ForwardDecayReservoir {
        ForwardDecayReservoir::from_size_and_alpha(DEFAULT_SIZE, DEFAULT_ALPHA)
    }

    pub fn from_size_and_alpha(size: usize, alpha: f64) -> ForwardDecayReservoir {
        let now = Instant::now();

        ForwardDecayReservoir {
            values: BTreeMap::new(),
            alpha: alpha,
            size: size,
            start_time: now,
            next_scale_time: now + Duration::from_secs(RESCALE_THRESHOLD_SECS),
        }
    }

    pub fn update(&mut self, value: i64) {
        self.update_at(Instant::now(), value);
    }

    pub fn update_at(&mut self, time: Instant, value: i64) {
        self.rescale_if_needed(time);

        let item_weight = self.weight(time - self.start_time);
        let sample = WeightedSample {
            value,
            weight: item_weight,
        };
        // Open01 since we don't want to divide by 0
        let priority = item_weight / rand::thread_rng().gen::<Open01<f64>>().0;
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

    fn rescale_if_needed(&mut self, now: Instant) {
        if now >= self.next_scale_time {
            self.rescale(now);
        }
    }

    pub fn snapshot(&self) -> WeightedSnapshot {
        let mut entries = self.values
            .values()
            .map(|s| {
                     SnapshotEntry {
                         value: s.value,
                         norm_weight: s.weight,
                         quantile: 0.,
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
            .fold(0., |acc, e| {
                e.quantile = acc;
                acc + e.norm_weight
            });

        WeightedSnapshot(entries)
    }

    fn weight(&self, time: Duration) -> f64 {
        (self.alpha * time.as_secs() as f64).exp()
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
    quantile: f64,
}

pub struct WeightedSnapshot(Vec<SnapshotEntry>);

impl WeightedSnapshot {
    pub fn value(&self, quantile: f64) -> i64 {
        assert!(quantile >= 0. && quantile <= 1.);

        if self.0.is_empty() {
            return 0;
        }

        let quantile = NotNaN::from(quantile);
        let idx = match self.0
                  .binary_search_by(|e| NotNaN::from(e.quantile).cmp(&quantile)) {
            Ok(idx) => idx,
            Err(idx) if idx >= self.0.len() => self.0.len() - 1,
            Err(idx) => idx,
        };

        self.0[idx].value
    }

    pub fn max(&self) -> i64 {
        self.0.last().map_or(0, |e| e.value)
    }

    pub fn min(&self) -> i64 {
        self.0.first().map_or(0, |e| e.value)
    }

    pub fn mean(&self) -> f64 {
        self.0
            .iter()
            .map(|e| e.value as f64 * e.norm_weight)
            .sum::<f64>()
    }

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
    fn a_histogram_of_100_out_of_1000_elements() {
        let mut histogram = ForwardDecayReservoir::from_size_and_alpha(100, 0.99);
        for i in 0..1000 {
            histogram.update(i);
        }

        assert_eq!(histogram.values.len(), 100);

        let snapshot = histogram.snapshot();

        assert_eq!(snapshot.0.len(), 100);

        assert_all_values_between(snapshot, 0..1000);
    }

    #[test]
    fn a_histogram_of_100_out_of_10_elements() {
        let mut histogram = ForwardDecayReservoir::from_size_and_alpha(100, 0.99);
        for i in 0..10 {
            histogram.update(i);
        }

        let snapshot = histogram.snapshot();

        assert_eq!(snapshot.0.len(), 10);

        assert_all_values_between(snapshot, 0..10);
    }

    #[test]
    fn a_heavily_biased_histogram_of_100_out_of_1000_elements() {
        let mut histogram = ForwardDecayReservoir::from_size_and_alpha(1000, 0.01);
        for i in 0..100 {
            histogram.update(i);
        }

        assert_eq!(histogram.values.len(), 100);

        let snapshot = histogram.snapshot();

        assert_eq!(snapshot.0.len(), 100);

        assert_all_values_between(snapshot, 0..100);
    }

    #[test]
    fn long_periods_of_inactivity_should_not_corrupt_sampling_state() {
        let mut histogram = ForwardDecayReservoir::from_size_and_alpha(10, 0.015);
        let mut now = histogram.start_time;

        // add 1000 values at a rate of 10 values/second
        let delta = Duration::from_millis(100);
        for i in 0..1000 {
            now += delta;
            histogram.update_at(now, 1000 + i);
        }

        let snapshot = histogram.snapshot();
        assert_eq!(snapshot.0.len(), 10);
        assert_all_values_between(snapshot, 1000..2000);

        // wait for 15 hours and add another value.
        // this should trigger a rescale. Note that the number of samples will
        // be reduced because of the very small scaling factor that will make
        // all existing priorities equal to zero after rescale.
        now += Duration::from_secs(15 * 60 * 60);
        histogram.update_at(now, 2000);

        let snapshot = histogram.snapshot();
        assert_eq!(snapshot.0.len(), 2);
        assert_all_values_between(snapshot, 1000..3000);

        // add 1000 values at a rate of 10 values/second
        for i in 0..1000 {
            now += delta;
            histogram.update_at(now, 3000 + i);
        }
        let snapshot = histogram.snapshot();
        assert_eq!(snapshot.0.len(), 10);
        assert_all_values_between(snapshot, 3000..4000);
    }

    #[test]
    fn spot_lift() {
        let mut histogram = ForwardDecayReservoir::from_size_and_alpha(1000, 0.015);
        let mut now = histogram.start_time;

        let values_per_minute = 10;
        let values_interval = Duration::from_secs(60) / values_per_minute;
        // mode 1: steady regime for 120 minutes
        for _ in 0..120 * values_per_minute {
            histogram.update_at(now, 177);
            now += values_interval;
        }

        // switching to mode 2: 10 minutes with the same rate, but larger value
        for _ in 0..10 * values_per_minute {
            histogram.update_at(now, 9999);
            now += values_interval;
        }

        // expect that the quantiles should be about mode 2 after 10 minutes
        assert_eq!(histogram.snapshot().value(0.5), 9999);
    }

    #[test]
    fn spot_fall() {
        let mut histogram = ForwardDecayReservoir::from_size_and_alpha(1000, 0.015);
        let mut now = histogram.start_time;

        let values_per_minute = 10;
        let values_interval = Duration::from_secs(60) / values_per_minute;
        // mode 1: steady regime for 120 minutes
        for _ in 0..120 * values_per_minute {
            histogram.update_at(now, 9998);
            now += values_interval;
        }

        // switching to mode 2: 10 minutes with the same rate, but smaller value
        for _ in 0..10 * values_per_minute {
            histogram.update_at(now, 178);
            now += values_interval;
        }

        // expect that the quantiles should be about mode 2 after 10 minutes
        assert_eq!(histogram.snapshot().value(0.5), 178);
    }

    #[test]
    fn quantiles_should_be_based_on_weights() {
        let mut histogram = ForwardDecayReservoir::from_size_and_alpha(1000, 0.015);
        let mut now = histogram.start_time;

        for _ in 0..40 {
            histogram.update_at(now, 177);
        }

        now += Duration::from_secs(120);

        for _ in 0..10 {
            histogram.update_at(now, 9999);
        }

        let snapshot = histogram.snapshot();
        assert_eq!(snapshot.0.len(), 50);

        // the first added 40 items (177) have weights 1
        // the next added 10 items (9999) have weights ~6
        // so, it's 40 vs 60 distribution, not 40 vs 10
        assert_eq!(snapshot.value(0.5), 9999);
        assert_eq!(snapshot.value(0.75), 9999);
    }

    fn assert_all_values_between(snapshot: WeightedSnapshot, range: Range<i64>) {
        for entry in &snapshot.0 {
            assert!(entry.value >= range.start && entry.value < range.end,
                    "snapshot value {} was not in {:?}",
                    entry.value,
                    range);
        }
    }
}
