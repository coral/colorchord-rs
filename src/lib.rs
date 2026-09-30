//! ColorChord audio analysis in pure Rust.
//!
//! Push normalized mono samples, then call [`NoteFinder::analyze`] at a fixed
//! cadence (normally 60 Hz). Analysis cadence affects smoothing and note identity.
//! Construction allocates and precomputes Quick coefficients; ingestion, analysis,
//! and reset do not allocate. The default `simd` feature selects a supported CPU
//! backend through Fearless SIMD. Disable default features for the scalar path.
//!
//! ```
//! use colorchord_rs::{Config, NoteFinder};
//! let mut finder = NoteFinder::new(Config::default())?;
//! finder.push_samples(&[0.0_f32; 735]);
//! let frame = finder.analyze();
//! assert!(frame.notes.iter().all(|n| !n.active()));
//! # Ok::<(), colorchord_rs::ConfigError>(())
//! ```
#![forbid(unsafe_code)]
#![warn(missing_docs)]
pub mod color;
mod config;
mod dft;
mod tracking;

pub use config::{Config, ConfigError, DftAlgorithm};
// These are the same traits CPAL re-exports, without platform audio dependencies.
pub use dasp_sample::{FromSample, Sample};

/// A spectral peak before temporal note tracking.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Distribution {
    /// Amplitude before output chopping.
    pub amplitude: f32,
    /// Circular position measured in bins within one octave.
    pub mean: f32,
    /// Distribution width metadata from the configuration.
    pub sigma: f32,
    /// Whether the tracker consumed this peak during the current frame.
    pub taken: bool,
}

/// A stable tracking slot. Inactive slots retain diagnostic history.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Note {
    /// Enduring identity; zero means the slot has no enduring identity.
    pub id: u32,
    /// Position in bins, not Hz. Inactive slots may contain sentinel values.
    pub position: f32,
    /// Amplitude before output chopping.
    pub amplitude: f32,
    /// Chopped amplitude, clamped to zero. Determines whether the note is active.
    pub amplitude_out: f32,
    /// Secondary smoothed amplitude, used for note retirement.
    pub amplitude_iir2: f32,
    /// Whether a distribution matched this slot during the current frame.
    pub matched: bool,
    /// Last attachment index, for diagnostics. It can be stale after reassignment.
    /// This is not a guaranteed association with the current distributions.
    pub distribution_index: usize,
}
impl Note {
    /// Whether this slot has a positive output amplitude.
    pub fn active(&self) -> bool {
        self.amplitude_out > 0.0
    }
    /// Normalize the position to one octave = 1.0.
    ///
    /// Pass `finder.config().bins_per_octave`; inactive slots can return sentinels.
    /// The bin count must be nonzero and match the finder that produced this note.
    pub fn pitch_class(&self, bins_per_octave: usize) -> f32 {
        self.position / bins_per_octave as f32
    }
}

/// Borrowed results, valid until the next mutable operation on the finder.
#[derive(Debug)]
pub struct Frame<'a> {
    /// Bin periods measured in samples.
    pub periods: &'a [f32],
    /// Unfiltered transform magnitudes.
    pub raw_bins: &'a [f32],
    /// Spectrum after gain, temporal smoothing, and edge tapering.
    pub spectrum: &'a [f32],
    /// One octave of bins after folding and circular filtering.
    pub folded: &'a [f32],
    /// Current peaks in ascending amplitude order.
    pub distributions: &'a [Distribution],
    /// Stable slot order, including inactive slots.
    pub notes: &'a [Note],
}

/// An independent, synchronous analysis pipeline with preallocated working storage.
pub struct NoteFinder {
    config: Config,
    transform: dft::Transform,
    periods: Vec<f32>,
    raw: Vec<f32>,
    spectrum: Vec<f32>,
    folded: Vec<f32>,
    scratch: Vec<f32>,
    distributions: Vec<Distribution>,
    distribution_count: usize,
    notes: Vec<Note>,
    next_id: u32,
}
impl NoteFinder {
    /// Validate configuration, allocate processing storage, and precompute coefficients.
    ///
    /// Quick coefficient rows use up to 8 MiB per finder (about 526 KiB with defaults
    /// and 5.21 MiB with desktop settings). Larger tables fall back to direct computation.
    /// Construct outside the realtime callback; processing and reset reuse this storage.
    pub fn new(config: Config) -> Result<Self, ConfigError> {
        config.validate()?;
        let bins = config.bins_per_octave * config.octaves;
        // Divide in f32 before promoting to f64 for exponentiation and division.
        let periods: Vec<f32> = (0..bins)
            .map(|i| {
                ((config.sample_rate as f32 / config.base_hz) as f64
                    / 2.0_f64.powf((i as f32 / config.bins_per_octave as f32) as f64))
                    as f32
            })
            .collect();
        let transform = dft::Transform::new(&config, &periods);
        let width = config.bins_per_octave;
        Ok(Self {
            transform,
            periods,
            raw: vec![0.0; bins],
            spectrum: vec![0.0; bins],
            folded: vec![0.0; width],
            scratch: vec![0.0; width],
            distributions: vec![Distribution::default(); width / 2],
            distribution_count: 0,
            notes: vec![Note::default(); width / 2],
            next_id: 1,
            config,
        })
    }
    /// Settings used by this finder; reconstruct it to change them.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Ingest mono PCM using CPAL's sample conversion conventions.
    /// Non-finite floats become silence; finite input is clamped to [-1, 1].
    /// An empty slice does nothing. Chunk boundaries do not affect the result.
    pub fn push_samples<S: Sample>(&mut self, samples: &[S])
    where
        f32: FromSample<S>,
    {
        self.transform.push_samples(samples);
    }

    /// Analyze the current history and advance note smoothing once.
    /// Quick wraps within the configured history.
    pub fn analyze(&mut self) -> Frame<'_> {
        self.transform.analyze(&mut self.raw);
        tracking::spectrum(
            &self.config,
            &self.raw,
            &mut self.spectrum,
            &mut self.folded,
            &mut self.scratch,
        );
        self.distribution_count =
            tracking::decompose(&self.config, &self.folded, &mut self.distributions);
        tracking::track(
            &self.config,
            &mut self.distributions[..self.distribution_count],
            &mut self.notes,
            &mut self.next_id,
        );
        self.frame()
    }

    /// Read the most recent analysis without advancing smoothing or tracking.
    pub fn frame(&self) -> Frame<'_> {
        Frame {
            periods: &self.periods,
            raw_bins: &self.raw,
            spectrum: &self.spectrum,
            folded: &self.folded,
            distributions: &self.distributions[..self.distribution_count],
            notes: &self.notes,
        }
    }

    /// Restore newly constructed state without allocating or changing configuration.
    pub fn reset(&mut self) {
        self.transform.reset();
        self.raw.fill(0.0);
        self.spectrum.fill(0.0);
        self.folded.fill(0.0);
        self.scratch.fill(0.0);
        self.distributions.fill(Distribution::default());
        self.distribution_count = 0;
        self.notes.fill(Note::default());
        self.next_id = 1;
    }
}
