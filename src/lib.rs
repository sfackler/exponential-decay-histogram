//! A histogram which exponentially weights in favor of recent values.
//!
//! Histograms compute statistics about the distribution of values in a data
//! set. This histogram exponentially favors recent values over older ones,
//! making it suitable for use cases such as monitoring the state of long
//! running processes.
//!
//! The histogram does not store all values simultaneously, but rather a
//! randomized subset. This allows us to put bounds on overall memory use
//! regardless of the rate of events.
//!
//! This implementation is based on the `ExponentiallyDecayingReservoir` class
//! in the Java [Metrics][1] library, which is itself based on the forward decay
//! model described in [Cormode et al. 2009][2].
//!
//! [1]: http://metrics.dropwizard.io/3.2.2/
//! [2]: http://dimacs.rutgers.edu/~graham/pubs/papers/fwddecay.pdf
//!
//! # Examples
//!
//! ```
//! use exponential_decay_histogram::ExponentialDecayHistogram;
//!
//! # fn do_work() -> i64 { 0 }
//! let mut histogram = ExponentialDecayHistogram::new();
//!
//! // Do some work for a while and fill the histogram with some information.
//! // Even though we're putting 10000 values into the histogram, it will only
//! // retain a subset of them.
//! for _ in 0..10000 {
//!     let size = do_work();
//!     histogram.update(size);
//! }
//!
//! // Take a snapshot to inspect the current state of the histogram.
//! let snapshot = histogram.snapshot();
//! println!("count: {}", snapshot.count());
//! println!("min: {}", snapshot.min());
//! println!("max: {}", snapshot.max());
//! println!("mean: {}", snapshot.mean());
//! println!("standard deviation: {}", snapshot.stddev());
//! println!("median: {}", snapshot.value(0.5));
//! println!("99th percentile: {}", snapshot.value(0.99));
//! ```
#![warn(missing_docs)]
use ordered_float::NotNan;
use rand::distr::Open01;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::iter;
use std::slice;
use std::time::{Duration, Instant};

const RESCALE_THRESHOLD: Duration = Duration::from_secs(60 * 60);

#[derive(Debug)]
struct WeightedSample {
    value: i64,
    weight: f64,
}

/// A histogram which exponentially weights in favor of recent values.
///
/// See the crate level documentation for more details.
#[derive(Debug)]
pub struct ExponentialDecayHistogram {
    values: BTreeMap<NotNan<f64>, WeightedSample>,
    alpha: f64,
    size: usize,
    count: u64,
    start_time: Instant,
    next_scale_time: Instant,
    rng: SmallRng,
}

impl Default for ExponentialDecayHistogram {
    fn default() -> ExponentialDecayHistogram {
        ExponentialDecayHistogram::new()
    }
}

impl ExponentialDecayHistogram {
    /// Returns a new histogram with a default configuration.
    pub fn new() -> Self {
        Self::builder().build()
    }

    /// Returns a new builder to create a histogram with a custom configuration.
    pub fn builder() -> Builder {
        Builder {
            now: Instant::now(),
            size: 1028,
            alpha: 0.015,
        }
    }

    /// Returns a new histogram configured with the specified size and alpha.
    ///
    /// `size` specifies the number of values stored in the histogram. A larger
    /// size will provide more accurate statistics, but with a higher memory
    /// overhead.
    ///
    /// `alpha` specifies the exponential decay factor. A larger factor biases
    /// the histogram towards newer values.
    ///
    /// # Panics
    ///
    /// Panics if `size` is zero.
    #[deprecated = "use `ExponentialDecayHistogram::builder()` instead"]
    pub fn with_size_and_alpha(size: usize, alpha: f64) -> Self {
        Self::builder().size(size).alpha(alpha).build()
    }

    /// Inserts a value into the histogram at the current time.
    pub fn update(&mut self, value: i64) {
        self.update_at(Instant::now(), value);
    }

    /// Inserts a value into the histogram at the specified time.
    ///
    /// # Panics
    ///
    /// May panic if values are inserted at non-monotonically increasing times.
    pub fn update_at(&mut self, time: Instant, value: i64) {
        self.rescale_if_needed(time);
        self.count += 1;

        let item_weight = self.weight(time);
        let sample = WeightedSample {
            value,
            weight: item_weight,
        };
        // Open01 since we don't want to divide by 0
        let priority = item_weight / self.rng.sample::<f64, _>(&Open01);
        let priority = NotNan::new(priority).unwrap();

        if self.values.len() < self.size {
            self.values.insert(priority, sample);
        } else {
            let first = *self.values.keys().next().unwrap();
            if first < priority && self.values.insert(priority, sample).is_none() {
                self.values.remove(&first).unwrap();
            }
        }
    }

    /// Takes a snapshot of the current state of the histogram.
    pub fn snapshot(&self) -> Snapshot {
        let mut entries = self
            .values
            .values()
            .map(|s| SnapshotEntry {
                value: s.value,
                norm_weight: s.weight,
                quantile: NotNan::new(0.).unwrap(),
            })
            .collect::<Vec<_>>();

        entries.sort_by_key(|e| e.value);

        let sum_weight = entries.iter().map(|e| e.norm_weight).sum::<f64>();
        for entry in &mut entries {
            entry.norm_weight /= sum_weight;
        }

        entries.iter_mut().fold(NotNan::new(0.).unwrap(), |acc, e| {
            e.quantile = acc;
            acc + NotNan::new(e.norm_weight).unwrap()
        });

        Snapshot {
            entries,
            count: self.count,
        }
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
        self.next_scale_time = now + RESCALE_THRESHOLD;
        let old_start_time = self.start_time;
        self.start_time = now;
        let scaling_factor =
            NotNan::new((-self.alpha * (now - old_start_time).as_secs() as f64).exp()).unwrap();

        self.values = self
            .values
            .iter()
            .map(|(&k, v)| {
                (
                    k * scaling_factor,
                    WeightedSample {
                        value: v.value,
                        weight: v.weight * *scaling_factor,
                    },
                )
            })
            .collect();
    }
}

/// A builder type for [`ExponentialDecayHistogram`] objects.
pub struct Builder {
    now: Instant,
    size: usize,
    alpha: f64,
}

impl Builder {
    /// Sets the construction time of the histogram.
    ///
    /// Defaults to the system time when the [`Builder`] was constructed.
    pub fn at(&mut self, now: Instant) -> &mut Self {
        self.now = now;
        self
    }

    /// Sets the size of the histogram.
    ///
    /// A larger size will provide a more accurate histogram, but with a higher memory overhead.
    ///
    /// Defaults to 1028, which offers a 99.9% confidence level with a 5% margin of error.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    pub fn size(&mut self, size: usize) -> &mut Self {
        assert!(size > 0);

        self.size = size;
        self
    }

    /// Sets the decay rate of the histogram.
    ///
    /// A larger value biases the histogram towards newer values.
    ///
    /// Defaults to 0.015, which heavily biases towards the last 5 minutes of values.
    pub fn alpha(&mut self, alpha: f64) -> &mut Self {
        self.alpha = alpha;
        self
    }

    /// Creates a new [`ExponentialDecayHistogram`].
    pub fn build(&self) -> ExponentialDecayHistogram {
        ExponentialDecayHistogram {
            values: BTreeMap::new(),
            alpha: self.alpha,
            size: self.size,
            count: 0,
            start_time: self.now,
            // we store this explicitly because it's ~10% faster than doing the math on demand
            next_scale_time: self.now + RESCALE_THRESHOLD,
            // using a SmallRng is ~10% faster than using thread_rng()
            rng: SmallRng::from_rng(&mut rand::rng()),
        }
    }
}

struct SnapshotEntry {
    value: i64,
    norm_weight: f64,
    quantile: NotNan<f64>,
}

/// A snapshot of the state of an `ExponentialDecayHistogram` at some point in time.
pub struct Snapshot {
    entries: Vec<SnapshotEntry>,
    count: u64,
}

impl Snapshot {
    /// Returns the value at a specified quantile in the snapshot, or 0 if it is
    /// empty.
    ///
    /// For example, `snapshot.value(0.5)` returns the median value of the
    /// snapshot.
    ///
    /// # Panics
    ///
    /// Panics if `quantile` is not between 0 and 1 (inclusive).
    pub fn value(&self, quantile: f64) -> i64 {
        assert!(quantile >= 0. && quantile <= 1.);

        if self.entries.is_empty() {
            return 0;
        }

        let quantile = NotNan::new(quantile).unwrap();
        let idx = match self.entries.binary_search_by(|e| e.quantile.cmp(&quantile)) {
            Ok(idx) => idx,
            Err(idx) if idx >= self.entries.len() => self.entries.len() - 1,
            Err(idx) => idx,
        };

        self.entries[idx].value
    }

    /// Returns the largest value in the snapshot, or 0 if it is empty.
    pub fn max(&self) -> i64 {
        self.entries.last().map_or(0, |e| e.value)
    }

    /// Returns the smallest value in the snapshot, or 0 if it is empty.
    pub fn min(&self) -> i64 {
        self.entries.first().map_or(0, |e| e.value)
    }

    /// Returns the mean of the values in the snapshot, or 0 if it is empty.
    pub fn mean(&self) -> f64 {
        self.entries
            .iter()
            .map(|e| e.value as f64 * e.norm_weight)
            .sum::<f64>()
    }

    /// Returns the standard deviation of the values in the snapshot, or 0 if it
    /// is empty.
    pub fn stddev(&self) -> f64 {
        if self.entries.len() <= 1 {
            return 0.;
        }

        let mean = self.mean();
        let variance = self
            .entries
            .iter()
            .map(|e| {
                let diff = e.value as f64 - mean;
                e.norm_weight * diff * diff
            })
            .sum::<f64>();

        variance.sqrt()
    }

    /// Returns the number of values which have been written to the histogram at
    /// the time of the snapshot.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns an iterator over the distinct values in the snapshot along with their weights.
    pub fn values<'a>(&'a self) -> Values<'a> {
        Values {
            it: self.entries.iter().peekable(),
        }
    }
}

/// An iterator over the distinct values in a snapshot along with their weights.
pub struct Values<'a> {
    it: iter::Peekable<slice::Iter<'a, SnapshotEntry>>,
}

impl<'a> Iterator for Values<'a> {
    type Item = (i64, f64);

    fn next(&mut self) -> Option<(i64, f64)> {
        let (value, mut weight) = match self.it.next() {
            Some(v) => (v.value, v.norm_weight),
            None => return None,
        };

        loop {
            match self.it.peek() {
                Some(v) if v.value == value => weight += v.norm_weight,
                _ => break,
            }
            self.it.next();
        }

        Some((value, weight))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::ops::Range;

    #[test]
    fn a_histogram_of_100_out_of_1000_elements() {
        let mut histogram = ExponentialDecayHistogram::builder()
            .size(100)
            .alpha(0.99)
            .build();
        for i in 0..1000 {
            histogram.update(i);
        }

        assert_eq!(histogram.values.len(), 100);

        let snapshot = histogram.snapshot();

        assert_eq!(snapshot.entries.len(), 100);

        assert_all_values_between(snapshot, 0..1000);
    }

    #[test]
    fn a_histogram_of_100_out_of_10_elements() {
        let mut histogram = ExponentialDecayHistogram::builder()
            .size(100)
            .alpha(0.99)
            .build();
        for i in 0..10 {
            histogram.update(i);
        }

        let snapshot = histogram.snapshot();

        assert_eq!(snapshot.entries.len(), 10);

        assert_all_values_between(snapshot, 0..10);
    }

    #[test]
    fn a_heavily_biased_histogram_of_100_out_of_1000_elements() {
        let mut histogram = ExponentialDecayHistogram::builder()
            .size(1000)
            .alpha(0.01)
            .build();
        for i in 0..100 {
            histogram.update(i);
        }

        assert_eq!(histogram.values.len(), 100);

        let snapshot = histogram.snapshot();

        assert_eq!(snapshot.entries.len(), 100);

        assert_all_values_between(snapshot, 0..100);
    }

    #[test]
    fn long_periods_of_inactivity_should_not_corrupt_sampling_state() {
        let mut now = Instant::now();
        let mut histogram = ExponentialDecayHistogram::builder()
            .at(now)
            .size(10)
            .alpha(0.015)
            .build();

        // add 1000 values at a rate of 10 values/second
        let delta = Duration::from_millis(100);
        for i in 0..1000 {
            now += delta;
            histogram.update_at(now, 1000 + i);
        }

        let snapshot = histogram.snapshot();
        assert_eq!(snapshot.entries.len(), 10);
        assert_all_values_between(snapshot, 1000..2000);

        // wait for 15 hours and add another value.
        // this should trigger a rescale. Note that the number of samples will
        // be reduced because of the very small scaling factor that will make
        // all existing priorities equal to zero after rescale.
        now += Duration::from_secs(15 * 60 * 60);
        histogram.update_at(now, 2000);

        let snapshot = histogram.snapshot();
        assert_eq!(snapshot.entries.len(), 2);
        assert_all_values_between(snapshot, 1000..3000);

        // add 1000 values at a rate of 10 values/second
        for i in 0..1000 {
            now += delta;
            histogram.update_at(now, 3000 + i);
        }
        let snapshot = histogram.snapshot();
        assert_eq!(snapshot.entries.len(), 10);
        assert_all_values_between(snapshot, 3000..4000);
    }

    #[test]
    fn spot_lift() {
        let mut now = Instant::now();
        let mut histogram = ExponentialDecayHistogram::builder()
            .at(now)
            .size(1000)
            .alpha(0.015)
            .build();

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
        let mut now = Instant::now();
        let mut histogram = ExponentialDecayHistogram::builder()
            .at(now)
            .size(1000)
            .alpha(0.015)
            .build();

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
        let mut now = Instant::now();
        let mut histogram = ExponentialDecayHistogram::builder()
            .at(now)
            .size(1000)
            .alpha(0.015)
            .build();

        for _ in 0..40 {
            histogram.update_at(now, 177);
        }

        now += Duration::from_secs(120);

        for _ in 0..10 {
            histogram.update_at(now, 9999);
        }

        let snapshot = histogram.snapshot();
        assert_eq!(snapshot.entries.len(), 50);

        // the first added 40 items (177) have weights 1
        // the next added 10 items (9999) have weights ~6
        // so, it's 40 vs 60 distribution, not 40 vs 10
        assert_eq!(snapshot.value(0.5), 9999);
        assert_eq!(snapshot.value(0.75), 9999);
    }

    fn assert_all_values_between(snapshot: Snapshot, range: Range<i64>) {
        for entry in &snapshot.entries {
            assert!(
                entry.value >= range.start && entry.value < range.end,
                "snapshot value {} was not in {:?}",
                entry.value,
                range
            );
        }
    }

    #[test]
    fn values() {
        let mut histogram = ExponentialDecayHistogram::new();
        let now = histogram.start_time;

        histogram.update_at(now, 1);
        histogram.update_at(now, 1);
        histogram.update_at(now, 1);
        histogram.update_at(now, 10);

        let values = histogram.snapshot().values().collect::<Vec<_>>();
        assert_eq!(values, vec![(1, 0.75), (10, 0.25)]);
    }
}
