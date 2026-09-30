use std::{error::Error, fmt};

/// Configurable floating-point and octave-skipping integer transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DftAlgorithm {
    /// Configurable floating-point transform with frequency-dependent strides.
    Quick,
    /// Fixed 24-bin, five-octave integer transform with octave scheduling.
    Progressive32,
}

/// Immutable analysis settings. Defaults use 24 bins per octave across five octaves.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Input mono sample rate in Hz (8,000–384,000).
    pub sample_rate: u32,
    /// Quick history capacity in samples (2–1,048,576); unused by Progressive32.
    pub history_size: usize,
    /// Transform implementation. All remaining tracking stages are shared.
    pub algorithm: DftAlgorithm,
    /// Number of octaves (1–8); Progressive32 requires five.
    pub octaves: usize,
    /// Even number of bins per octave (12–128); Progressive32 requires 24.
    pub bins_per_octave: usize,
    /// Lowest bin frequency in Hz. The highest bin must remain below Nyquist.
    pub base_hz: f32,
    /// Circular neighbor smoothing weight (0–1).
    pub filter_strength: f32,
    /// Number of circular smoothing passes (0–32).
    pub filter_iterations: usize,
    /// Spectrum gain before folding and compression (0–100).
    pub amplification: f32,
    /// Distribution normalization coefficient (1e-6–100).
    pub compression_coefficient: f32,
    /// Distribution normalization exponent (0–1).
    pub compression_exponent: f32,
    /// Quick stride divisor (1–10,000,000); larger values inspect more samples.
    pub dft_speedup: f32,
    /// Positive Quick window length measured in cycles of each bin frequency.
    pub dft_q: f32,
    /// Weight of the previous spectrum per analysis call (0–1).
    pub dft_iir: f32,
    /// Additional gain per bin index (0–100).
    pub slope: f32,
    /// Positive distribution width metadata; turbo decomposition does not fit it.
    pub default_sigma: f32,
    /// Maximum circular bin distance for attaching a peak (0–100).
    pub note_jumpability: f32,
    /// Maximum circular bin distance for merging notes (0–100).
    pub note_combine_distance: f32,
    /// New pitch weight per analysis call (0–1).
    pub note_attach_frequency_iir: f32,
    /// New amplitude weight per analysis call (0–1).
    pub note_attach_amplitude_iir: f32,
    /// Secondary amplitude smoothing weight (0–1).
    pub note_attach_amplitude_iir2: f32,
    /// Threshold for retiring or reassigning a note (0–100).
    pub note_min_amplitude: f32,
    /// Minimum distribution amplitude for assignment (0–100).
    pub note_minimum_new_distribution: f32,
    /// Amount subtracted from public output amplitudes (0–100).
    pub note_out_chop: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            history_size: 8096,
            algorithm: DftAlgorithm::Quick,
            octaves: 5,
            bins_per_octave: 24,
            base_hz: 55.0,
            filter_strength: 0.5,
            filter_iterations: 1,
            amplification: 1.0,
            compression_coefficient: 1.0,
            compression_exponent: 0.5,
            dft_speedup: 300.0,
            dft_q: 16.0,
            dft_iir: 0.0,
            slope: 0.0,
            default_sigma: 1.4,
            note_jumpability: 2.5,
            note_combine_distance: 0.5,
            note_attach_frequency_iir: 0.4,
            note_attach_amplitude_iir: 0.2,
            note_attach_amplitude_iir2: 0.05,
            note_min_amplitude: 0.001,
            note_minimum_new_distribution: 0.02,
            note_out_chop: 0.1,
        }
    }
}

impl Config {
    /// Higher-resolution desktop settings: 64 bins per octave across six octaves.
    pub fn desktop() -> Self {
        Self {
            amplification: 10.0,
            dft_iir: 0.6,
            dft_q: 60.0,
            dft_speedup: 1000.0,
            octaves: 6,
            bins_per_octave: 64,
            filter_iterations: 2,
            note_attach_amplitude_iir: 0.35,
            note_attach_amplitude_iir2: 0.25,
            note_attach_frequency_iir: 0.3,
            note_jumpability: 1.8,
            note_out_chop: 0.05,
            ..Self::default()
        }
    }

    /// Validate settings without allocating processing storage.
    ///
    /// Returns the first invalid field and its constraint. Construction calls
    /// this automatically. All floating-point settings must be finite.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let fail = |field, reason| Err(ConfigError { field, reason });
        if !(8000..=384_000).contains(&self.sample_rate) {
            return fail("sample_rate", "must be 8000..=384000");
        }
        if !(2..=1_048_576).contains(&self.history_size) {
            return fail("history_size", "must be 2..=1048576");
        }
        if !(1..=8).contains(&self.octaves) {
            return fail("octaves", "must be 1..=8");
        }
        if !(12..=128).contains(&self.bins_per_octave) || !self.bins_per_octave.is_multiple_of(2) {
            return fail("bins_per_octave", "must be even and 12..=128");
        }
        if self.algorithm == DftAlgorithm::Progressive32
            && (self.octaves != 5 || self.bins_per_octave != 24)
        {
            return fail(
                "algorithm",
                "Progressive32 requires 5 octaves and 24 bins per octave",
            );
        }
        for (name, value) in [
            ("base_hz", self.base_hz),
            ("dft_q", self.dft_q),
            ("dft_speedup", self.dft_speedup),
            ("default_sigma", self.default_sigma),
            ("compression_coefficient", self.compression_coefficient),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return fail(name, "must be finite and positive");
            }
        }
        for (name, value) in [
            ("filter_strength", self.filter_strength),
            ("dft_iir", self.dft_iir),
            ("compression_exponent", self.compression_exponent),
            ("note_attach_frequency_iir", self.note_attach_frequency_iir),
            ("note_attach_amplitude_iir", self.note_attach_amplitude_iir),
            (
                "note_attach_amplitude_iir2",
                self.note_attach_amplitude_iir2,
            ),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return fail(name, "must be finite and in 0..=1");
            }
        }
        for (name, value) in [
            ("amplification", self.amplification),
            ("slope", self.slope),
            ("note_jumpability", self.note_jumpability),
            ("note_combine_distance", self.note_combine_distance),
            ("note_min_amplitude", self.note_min_amplitude),
            (
                "note_minimum_new_distribution",
                self.note_minimum_new_distribution,
            ),
            ("note_out_chop", self.note_out_chop),
        ] {
            if !value.is_finite() || !(0.0..=100.0).contains(&value) {
                return fail(name, "must be finite and in 0..=100");
            }
        }
        if !(1e-6..=100.0).contains(&self.compression_coefficient) {
            return fail("compression_coefficient", "must be 1e-6..=100");
        }
        if !(1.0..=10_000_000.0).contains(&self.dft_speedup) {
            return fail("dft_speedup", "must be 1..=10000000");
        }
        if self.filter_iterations > 32 {
            return fail("filter_iterations", "must be <= 32");
        }
        // Keep float-to-integer conversions and Quick's loop bounded.
        let period = self.sample_rate as f32 / self.base_hz;
        let window = period * self.dft_q;
        if !window.is_finite() || !(1.0..=10_000_000.0).contains(&window) {
            return fail(
                "dft_q",
                "base period times dft_q must be 1..=10000000 samples",
            );
        }
        let highest = self.base_hz as f64
            * 2.0_f64.powf(
                (self.octaves * self.bins_per_octave - 1) as f64 / self.bins_per_octave as f64,
            );
        if highest >= self.sample_rate as f64 / 2.0 {
            return fail("base_hz", "highest bin must be below Nyquist");
        }
        Ok(())
    }
}

/// An invalid configuration field and the constraint it violates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConfigError {
    field: &'static str,
    reason: &'static str,
}
impl ConfigError {
    /// Configuration field (or algorithm combination) that failed validation.
    pub fn field(&self) -> &'static str {
        self.field
    }
    /// Constraint violated by the field.
    pub fn reason(&self) -> &'static str {
        self.reason
    }
}
impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid ColorChord setting {}: {}",
            self.field, self.reason
        )
    }
}
impl Error for ConfigError {}
