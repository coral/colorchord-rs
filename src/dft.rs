// Algorithms derived from ColorChord, Copyright 2015 Charles Lohr.
// See LICENSE-COLORCHORD. Constants and arithmetic order determine tracking behavior.
use crate::{Config, DftAlgorithm, FromSample, Sample};
#[cfg(feature = "simd")]
use fearless_simd::{Level, dispatch, f32x8, i32x8, prelude::*, u16x8};
use rtrb::{Consumer, Producer, RingBuffer};

// Algorithm-specific state is owned by its variant. Progressive32 does not need
// Quick's history, scratch copy, or per-sample ring-buffer operations.
pub(crate) enum Transform {
    Quick(Quick),
    Progressive32(Box<Progressive32>),
}

impl Transform {
    pub fn new(config: &Config, periods: &[f32]) -> Self {
        match config.algorithm {
            DftAlgorithm::Quick => Self::Quick(Quick::new(config, periods)),
            DftAlgorithm::Progressive32 => {
                Self::Progressive32(Box::new(Progressive32::new(periods)))
            }
        }
    }

    pub fn push_samples<S: Sample>(&mut self, samples: &[S])
    where
        f32: FromSample<S>,
    {
        // Dispatch once per slice, rather than branching on the algorithm per sample.
        let normalized = samples.iter().copied().map(|sample| {
            let sample = f32::from_sample(sample);
            if sample.is_finite() {
                sample.clamp(-1.0, 1.0)
            } else {
                0.0
            }
        });
        match self {
            Self::Quick(dft) => dft.push_samples(normalized),
            Self::Progressive32(dft) => dft.push_samples(normalized),
        }
    }

    pub fn analyze(&mut self, bins: &mut [f32]) {
        match self {
            Self::Quick(dft) => dft.analyze(bins),
            Self::Progressive32(dft) => dft.analyze(bins),
        }
    }

    #[cfg(all(test, feature = "simd"))]
    #[allow(dead_code)] // Used by the standalone benchmark.
    pub fn set_level(&mut self, level: Level) {
        match self {
            Self::Quick(dft) => dft.level = level,
            Self::Progressive32(dft) => dft.level = level,
        }
    }

    #[cfg(test)]
    #[allow(dead_code)] // Used by the standalone benchmark.
    pub fn coefficient_table_bytes(&self) -> usize {
        match self {
            Self::Quick(dft) => dft.cache.as_ref().map_or(0, |groups| {
                groups
                    .iter()
                    .map(|g| g.rows.len() * size_of::<QuickRow>())
                    .sum()
            }),
            Self::Progressive32(_) => 0,
        }
    }

    pub fn reset(&mut self) {
        match self {
            Self::Quick(dft) => dft.reset(),
            Self::Progressive32(dft) => dft.reset(),
        }
    }
}

pub(crate) struct Quick {
    bins: Box<[QuickBin]>,
    producer: Producer<f32>,
    consumer: Consumer<f32>,
    history: Box<[f32]>,
    q: f32,
    cache: Option<Box<[QuickGroup]>>,
    #[cfg(feature = "simd")]
    level: Level,
}

impl Quick {
    fn new(config: &Config, periods: &[f32]) -> Self {
        let (mut producer, consumer) = RingBuffer::new(config.history_size);
        for _ in 0..config.history_size {
            producer.push(0.0).expect("new history has capacity");
        }
        let bins: Box<[_]> = periods.iter().map(|&p| QuickBin::new(p, config)).collect();
        let cache = QuickGroup::build(&bins, config.history_size);
        Self {
            bins,
            producer,
            consumer,
            history: vec![0.0; config.history_size].into_boxed_slice(),
            q: config.dft_q,
            cache,
            #[cfg(feature = "simd")]
            level: Level::new(),
        }
    }

    fn push_samples(&mut self, mut samples: impl ExactSizeIterator<Item = f32>) {
        // Publish/retire chunks instead of doing two atomic queue operations per sample.
        while samples.len() > 0 {
            let count = samples.len().min(self.history.len());
            self.consumer
                .read_chunk(count)
                .expect("history is full")
                .commit_all();
            self.producer
                .write_chunk_uninit(count)
                .expect("retired history slots are free")
                .fill_from_iter(samples.by_ref().take(count));
        }
    }

    fn analyze(&mut self, output: &mut [f32]) {
        let chunk = self
            .consumer
            .read_chunk(self.history.len())
            .expect("history is full");
        let (a, b) = chunk.as_slices();
        self.history[..a.len()].copy_from_slice(a);
        self.history[a.len()..].copy_from_slice(b);
        if let Some(groups) = &self.cache {
            #[cfg(feature = "simd")]
            dispatch!(self.level, simd => analyze_cached_simd(simd, groups, &self.bins, &self.history, self.q, output));
            #[cfg(not(feature = "simd"))]
            analyze_cached_scalar(groups, &self.bins, &self.history, self.q, output);
        } else {
            for (out, bin) in output.iter_mut().zip(&self.bins) {
                *out = bin.analyze(&self.history, self.q);
            }
        }
    }

    fn reset(&mut self) {
        let mut chunk = self
            .consumer
            .read_chunk(self.history.len())
            .expect("history is full");
        let (a, b) = chunk.as_mut_slices();
        a.fill(0.0);
        b.fill(0.0);
        self.history.fill(0.0);
    }
}

struct QuickBin {
    period: f32,
    window: usize,
    skip: usize,
    advance: f32,
}
impl QuickBin {
    #[allow(clippy::approx_constant)] // Retain the established DFT phase increment.
    fn new(period: f32, config: &Config) -> Self {
        let window = (period * config.dft_q) as usize;
        let skip = ((window as f32 / config.dft_speedup).floor() as usize).max(1);
        let advance = (3.14159_f64 * 2.0 / period as f64) as f32 * skip as f32;
        Self {
            period,
            window,
            skip,
            advance,
        }
    }
    fn analyze(&self, history: &[f32], q: f32) -> f32 {
        let mut phase = 0.0_f32;
        let mut sine = 0.0_f32;
        let mut cosine = 0.0_f32;
        let mut position = history.len() - 1;
        let step = self.skip % history.len();
        for _ in (0..=self.window).step_by(self.skip) {
            let sample = history[position];
            sine += phase.sin() * sample;
            cosine += phase.cos() * sample;
            phase += self.advance;
            position = retreat(position, step, history.len());
        }
        (sine * sine + cosine * cosine).sqrt() / self.period / q * self.skip as f32
    }
}

// Vectorize across independent bins, preserving the addition order inside each bin.
// A sample-wise dot-product reduction would change rounding and note identities.
const QUICK_LANES: usize = 8;
const MAX_QUICK_CACHE_BYTES: usize = 8 * 1024 * 1024;

struct QuickRow {
    sine: [f32; QUICK_LANES],
    cosine: [f32; QUICK_LANES],
    index: [u32; QUICK_LANES],
}
struct QuickGroup {
    rows: Box<[QuickRow]>,
}
impl QuickGroup {
    fn build(bins: &[QuickBin], history_len: usize) -> Option<Box<[Self]>> {
        let iterations = |group: &[QuickBin]| {
            group
                .iter()
                .map(|b| b.window / b.skip + 1)
                .max()
                .unwrap_or(0)
        };
        let row_count: usize = bins.chunks(QUICK_LANES).map(iterations).sum();
        if row_count > MAX_QUICK_CACHE_BYTES / size_of::<QuickRow>() {
            return None;
        }
        Some(
            bins.chunks(QUICK_LANES)
                .map(|group| {
                    let mut phase = [0.0_f32; QUICK_LANES];
                    let mut position = [history_len - 1; QUICK_LANES];
                    let rows = (0..iterations(group))
                        .map(|iteration| {
                            let mut row = QuickRow {
                                sine: [0.0; QUICK_LANES],
                                cosine: [0.0; QUICK_LANES],
                                index: [0; QUICK_LANES],
                            };
                            for (lane, bin) in group.iter().enumerate() {
                                if iteration <= bin.window / bin.skip {
                                    row.sine[lane] = phase[lane].sin();
                                    row.cosine[lane] = phase[lane].cos();
                                    row.index[lane] = position[lane] as u32;
                                    phase[lane] += bin.advance;
                                    position[lane] = retreat(
                                        position[lane],
                                        bin.skip % history_len,
                                        history_len,
                                    );
                                }
                            }
                            row
                        })
                        .collect();
                    Self { rows }
                })
                .collect(),
        )
    }
}

#[inline]
fn retreat(position: usize, step: usize, len: usize) -> usize {
    if position < step {
        position + len - step
    } else {
        position - step
    }
}

#[cfg(any(not(feature = "simd"), test))]
fn analyze_cached_scalar(
    groups: &[QuickGroup],
    bins: &[QuickBin],
    history: &[f32],
    q: f32,
    output: &mut [f32],
) {
    for ((group, bins), output) in groups
        .iter()
        .zip(bins.chunks(QUICK_LANES))
        .zip(output.chunks_mut(QUICK_LANES))
    {
        let mut sine = [0.0; QUICK_LANES];
        let mut cosine = [0.0; QUICK_LANES];
        for row in &group.rows {
            for lane in 0..QUICK_LANES {
                let sample = history[row.index[lane] as usize];
                sine[lane] += row.sine[lane] * sample;
                cosine[lane] += row.cosine[lane] * sample;
            }
        }
        for (lane, (out, bin)) in output.iter_mut().zip(bins).enumerate() {
            *out = (sine[lane] * sine[lane] + cosine[lane] * cosine[lane]).sqrt() / bin.period / q
                * bin.skip as f32;
        }
    }
}

#[cfg(feature = "simd")]
#[inline(always)] // Fearless dispatch supplies the target-feature context.
fn analyze_cached_simd<S: Simd>(
    simd: S,
    groups: &[QuickGroup],
    bins: &[QuickBin],
    history: &[f32],
    q: f32,
    output: &mut [f32],
) {
    for ((group, bins), output) in groups
        .iter()
        .zip(bins.chunks(QUICK_LANES))
        .zip(output.chunks_mut(QUICK_LANES))
    {
        let mut sine = f32x8::splat(simd, 0.0);
        let mut cosine = f32x8::splat(simd, 0.0);
        for row in &group.rows {
            let samples = f32x8::from_fn(simd, |lane| history[row.index[lane] as usize]);
            sine += f32x8::from_slice(simd, &row.sine) * samples;
            cosine += f32x8::from_slice(simd, &row.cosine) * samples;
        }
        let amplitudes = (sine * sine + cosine * cosine).sqrt();
        for (lane, (out, bin)) in output.iter_mut().zip(bins).enumerate() {
            *out = amplitudes[lane] / bin.period / q * bin.skip as f32;
        }
    }
}

const BINS_PER_OCTAVE: usize = 24;
const OCTAVES: usize = 5;
const BINS: usize = BINS_PER_OCTAVE * OCTAVES;
const CYCLE: u8 = 1 << OCTAVES;
const IIR_SHIFT: u32 = 6;

pub(crate) struct Progressive32 {
    advances: [u16; BINS_PER_OCTAVE],
    phases: [u16; BINS],
    sine: [i32; BINS],
    cosine: [i32; BINS],
    output: [[i32; 2]; BINS],
    octave_sums: [i32; OCTAVES],
    schedule_position: u8,
    pending: Option<f32>,
    #[cfg(feature = "simd")]
    level: Level,
}
impl Progressive32 {
    pub fn new(periods: &[f32]) -> Self {
        let advances = std::array::from_fn(|i| {
            (65536.0_f64 / periods[i + BINS_PER_OCTAVE * (OCTAVES - 1)] as f64) as u16
        });
        Self {
            advances,
            phases: [0; BINS],
            sine: [0; BINS],
            cosine: [0; BINS],
            output: [[0; 2]; BINS],
            octave_sums: [0; OCTAVES],
            schedule_position: 0,
            pending: None,
            #[cfg(feature = "simd")]
            level: Level::new(),
        }
    }
    pub fn reset(&mut self) {
        self.phases.fill(0);
        self.sine.fill(0);
        self.cosine.fill(0);
        self.output.fill([0; 2]);
        self.octave_sums.fill(0);
        self.schedule_position = 0;
        self.pending = None;
    }
    fn push_samples(&mut self, samples: impl Iterator<Item = f32>) {
        // One runtime dispatch per input slice, never per sample or scheduler tick.
        #[cfg(feature = "simd")]
        dispatch!(self.level, simd => self.push_simd(simd, samples));
        #[cfg(not(feature = "simd"))]
        self.push_scalar(samples);
    }
    #[cfg(any(not(feature = "simd"), test))]
    fn push_scalar(&mut self, samples: impl Iterator<Item = f32>) {
        for sample in samples {
            // The analysis cursor excludes the latest sample; retain it until the next input.
            if let Some(previous) = self.pending.replace(sample) {
                let integer = (previous * 4095.0) as i16;
                for _ in 0..2 {
                    if let Some((octave, filtered)) = self.advance_tick(integer) {
                        for i in 0..BINS_PER_OCTAVE {
                            let bin = octave * BINS_PER_OCTAVE + i;
                            let phase = (self.phases[bin] >> 8) as usize;
                            self.phases[bin] = self.phases[bin].wrapping_add(self.advances[i]);
                            self.sine[bin] =
                                self.sine[bin].wrapping_add(SINE[phase] as i32 * filtered as i32);
                            self.cosine[bin] = self.cosine[bin]
                                .wrapping_add(SINE[(phase + 64) & 255] as i32 * filtered as i32);
                        }
                    }
                }
            }
        }
    }
    #[cfg(feature = "simd")]
    #[inline(always)]
    fn push_simd<S: Simd>(&mut self, simd: S, samples: impl Iterator<Item = f32>) {
        for sample in samples {
            if let Some(previous) = self.pending.replace(sample) {
                let integer = (previous * 4095.0) as i16;
                for _ in 0..2 {
                    if let Some((octave, filtered)) = self.advance_tick(integer) {
                        let sample = i32x8::splat(simd, filtered as i32);
                        for i in (0..BINS_PER_OCTAVE).step_by(8) {
                            let bin = octave * BINS_PER_OCTAVE + i;
                            let phase = u16x8::from_slice(simd, &self.phases[bin..bin + 8]);
                            let advances = u16x8::from_slice(simd, &self.advances[i..i + 8]);
                            (phase + advances).store_slice(&mut self.phases[bin..bin + 8]);
                            let index = phase >> 8;
                            let sine =
                                i32x8::from_fn(simd, |lane| SINE[index[lane] as usize] as i32);
                            let cosine = i32x8::from_fn(simd, |lane| {
                                SINE[(index[lane] as usize + 64) & 255] as i32
                            });
                            // SIMD integer operations preserve wrapping accumulator arithmetic.
                            (i32x8::from_slice(simd, &self.sine[bin..bin + 8]) + sine * sample)
                                .store_slice(&mut self.sine[bin..bin + 8]);
                            (i32x8::from_slice(simd, &self.cosine[bin..bin + 8]) + cosine * sample)
                                .store_slice(&mut self.cosine[bin..bin + 8]);
                        }
                    }
                }
            }
        }
    }
    #[inline(always)]
    fn advance_tick(&mut self, sample: i16) -> Option<(usize, i16)> {
        let slot = self.schedule_position;
        self.schedule_position = (slot + 1) & (CYCLE - 1);
        for sum in &mut self.octave_sums {
            *sum = sum.wrapping_add(sample as i32);
        }
        if slot == 0 {
            for ((output, sine), cosine) in self
                .output
                .iter_mut()
                .zip(&mut self.sine)
                .zip(&mut self.cosine)
            {
                *output = [*sine, *cosine];
                *sine = sine.wrapping_sub(*sine >> IIR_SHIFT);
                *cosine = cosine.wrapping_sub(*cosine >> IIR_SHIFT);
            }
            None
        } else {
            let octave = OCTAVES - 1 - (slot - 1).trailing_ones() as usize;
            let filtered = (self.octave_sums[octave] >> (OCTAVES - octave)) as i16;
            self.octave_sums[octave] = 0;
            Some((octave, filtered))
        }
    }
    pub fn analyze(&self, bins: &mut [f32]) {
        for (i, bin) in bins.iter_mut().enumerate() {
            let s = self.output[i][0].wrapping_abs() as f32;
            let c = self.output[i][1].wrapping_abs() as f32;
            *bin = ((s * s + c * c).sqrt() as f64 / 65536.0) as f32;
            *bin /= ((78 << IIR_SHIFT) * (1 << (i / BINS_PER_OCTAVE))) as f32;
        }
    }
}

// This lookup table is deliberately asymmetric; do not regenerate it.
const SINE: [i16; 256] = [
    0, 36, 73, 110, 147, 183, 220, 256, 292, 328, 364, 400, 435, 470, 505, 539, 574, 607, 641, 674,
    707, 739, 771, 802, 833, 863, 893, 922, 951, 979, 1007, 1034, 1060, 1086, 1111, 1135, 1159,
    1182, 1204, 1226, 1247, 1267, 1286, 1305, 1322, 1339, 1355, 1371, 1385, 1399, 1412, 1424, 1435,
    1445, 1455, 1463, 1471, 1477, 1483, 1488, 1492, 1495, 1498, 1499, 1500, 1499, 1498, 1495, 1492,
    1488, 1483, 1477, 1471, 1463, 1455, 1445, 1435, 1424, 1412, 1399, 1385, 1371, 1356, 1339, 1322,
    1305, 1286, 1267, 1247, 1226, 1204, 1182, 1159, 1135, 1111, 1086, 1060, 1034, 1007, 979, 951,
    922, 893, 863, 833, 802, 771, 739, 707, 674, 641, 607, 574, 539, 505, 470, 435, 400, 364, 328,
    292, 256, 220, 183, 147, 110, 73, 36, 0, -36, -73, -110, -146, -183, -219, -256, -292, -328,
    -364, -399, -435, -470, -505, -539, -573, -607, -641, -674, -706, -739, -771, -802, -833, -863,
    -893, -922, -951, -979, -1007, -1034, -1060, -1086, -1111, -1135, -1159, -1182, -1204, -1226,
    -1247, -1267, -1286, -1305, -1322, -1339, -1355, -1371, -1385, -1399, -1412, -1424, -1435,
    -1445, -1454, -1463, -1471, -1477, -1483, -1488, -1492, -1495, -1498, -1499, -1500, -1499,
    -1498, -1495, -1492, -1488, -1483, -1477, -1471, -1463, -1455, -1445, -1435, -1424, -1412,
    -1399, -1385, -1371, -1356, -1339, -1322, -1305, -1286, -1267, -1247, -1226, -1204, -1182,
    -1159, -1135, -1111, -1086, -1060, -1034, -1007, -979, -951, -923, -893, -863, -833, -802,
    -771, -739, -707, -674, -641, -608, -574, -540, -505, -470, -435, -400, -364, -328, -292, -256,
    -220, -183, -147, -110, -73, -37,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "simd")]
    fn supported_levels() -> Vec<Level> {
        let best = Level::new();
        #[allow(unused_mut)]
        let mut levels = vec![best];
        #[cfg(target_arch = "x86_64")]
        {
            levels.extend(best.as_sse2().map(Level::Sse2));
            levels.extend(best.as_sse4_2().map(Level::Sse4_2));
            levels.extend(best.as_avx2().map(Level::Avx2));
            levels.extend(best.as_avx512().map(Level::Avx512));
        }
        levels
    }

    #[test]
    fn cached_quick_matches_ordered_trig_for_tails_and_wrapped_history() {
        for config in [
            Config::default(),
            Config::desktop(),
            Config {
                bins_per_octave: 12,
                octaves: 1,
                history_size: 13,
                ..Config::default()
            },
        ] {
            let finder = crate::NoteFinder::new(config.clone()).unwrap();
            let bins: Vec<_> = finder
                .frame()
                .periods
                .iter()
                .map(|&p| QuickBin::new(p, &config))
                .collect();
            let groups = QuickGroup::build(&bins, config.history_size).unwrap();
            let history: Vec<_> = (0..config.history_size)
                .map(|i| (i as f32 * 0.37).sin())
                .collect();
            let expected: Vec<_> = bins
                .iter()
                .map(|b| b.analyze(&history, config.dft_q))
                .collect();
            let mut actual = vec![0.0; bins.len()];
            analyze_cached_scalar(&groups, &bins, &history, config.dft_q, &mut actual);
            assert_eq!(expected, actual);
            #[cfg(feature = "simd")]
            for level in supported_levels() {
                dispatch!(level, simd => analyze_cached_simd(simd, &groups, &bins, &history, config.dft_q, &mut actual));
                assert_eq!(expected, actual, "{level:?}");
            }
        }
    }

    #[test]
    fn quick_cache_budget_falls_back_without_building_large_tables() {
        let bin = QuickBin {
            period: 100.0,
            window: 1_000_000,
            skip: 1,
            advance: 0.0628,
        };
        assert!(QuickGroup::build(&[bin], 8096).is_none());
    }

    #[test]
    #[cfg(feature = "simd")]
    fn progressive_backends_match_scalar_through_wraps_and_chunk_boundaries() {
        let finder = crate::NoteFinder::new(Config::default()).unwrap();
        let samples: Vec<_> = (0..8193)
            .map(|i| match i % 4 {
                0 => -1.0,
                1 => 1.0,
                2 => 0.0,
                _ => (i as f32 * 0.13).sin(),
            })
            .collect();
        for level in supported_levels() {
            let mut scalar = Progressive32::new(finder.frame().periods);
            let mut vector = Progressive32::new(finder.frame().periods);
            vector.level = level;
            for state in [&mut scalar, &mut vector] {
                state.phases.fill(65530);
                state.sine = std::array::from_fn(|i| if i % 2 == 0 { i32::MAX } else { i32::MIN });
                state.cosine =
                    std::array::from_fn(|i| if i % 2 == 0 { i32::MIN } else { i32::MAX });
                state.schedule_position = 1; // Exercise wrapping before the first decay.
            }
            for chunk_size in [1, 31, 735, 8193] {
                for chunk in samples.chunks(chunk_size) {
                    scalar.push_scalar(chunk.iter().copied());
                    vector.push_samples(chunk.iter().copied());
                    vector.push_samples(std::iter::empty());
                    assert_eq!(scalar.phases, vector.phases, "{level:?}");
                    assert_eq!(scalar.sine, vector.sine, "{level:?}");
                    assert_eq!(scalar.cosine, vector.cosine, "{level:?}");
                    assert_eq!(scalar.output, vector.output, "{level:?}");
                    assert_eq!(scalar.octave_sums, vector.octave_sums);
                    assert_eq!(scalar.pending, vector.pending);
                    assert_eq!(scalar.schedule_position, vector.schedule_position);
                }
                scalar.reset();
                vector.reset();
            }
        }
    }
}
