# colorchord-rs

One of my personal inspirations is [cnlohr](https://github.com/cnlohr), a true C wizard that continues to push hardware in ways that inspires me. When onboarding onto Rust I wrapped their library [colorchord](https://github.com/cnlohr/colorchord) into [rustchord](https://github.com/coral/rustchord) which taught me the beauty of `bindgen` and honestly made me quit my previous job LOL. Well, a lot has changed since 2020. While I've had the pleasure & displeasure of writing tons of Rust the truth is the slop machine is getting quite good.

As I refreshed another project that used *rustchord*, I figured *"lets just port this straight to Rust"* after seeing the effort on [ColorChord.NET](https://www.colorchord.net/) so yeah this is a straight port to Rust. WHen developing I setup a test harness that compared the output to make sure the math matched and once we had parity I pushed the agent to optimize the functions to gain more performance.

Result of this is a more performant Colorchord with audio analysis, note tracking, and pitch-to-color conversion.
Includes configurable **Quick DFT** and octave-scheduled **Progressive32 DFT** transforms. Processing is synchronous and allocates nothing after construction.

## Use

```rust
use colorchord_rs::{Config, NoteFinder, color::note_to_rgb};

fn main() -> Result<(), colorchord_rs::ConfigError> {
    let config = Config::default(); // 44.1 kHz, Quick, 24 bins × 5 octaves
    let width = config.bins_per_octave;
    let mut finder = NoteFinder::new(config)?;

    // Supply mono PCM. At 44.1 kHz, 735 samples gives 60 analysis frames/s.
    finder.push_samples(&[0.0_f32; 735]);
    let frame = finder.analyze();
    for note in frame.notes.iter().filter(|note| note.active()) {
        let pitch = note.pitch_class(width);
        let rgb = note_to_rgb(pitch, 1.0, note.amplitude_out.min(1.0));
        // Consume note.id, pitch, amplitude, and rgb here.
    }
    Ok(())
}
```

`push_samples` accepts CPAL's `Sample`/`FromSample` traits, re-exported from
`dasp_sample` without device backends. Integer PCM is normalized through those
traits. Float input is clamped to [-1, 1]; non-finite values become silence.
Downmix interleaved audio before ingestion.

Chunk size is independent of analysis cadence. Call `analyze` at a fixed cadence:
note smoothing advances once per call, even when no new audio arrives. `frame`
reads the latest results without advancing state. Output slices borrow the
finder and include inactive note slots; filter with `Note::active` when needed.
Note identity is separate from pitch. Positions are octave-relative bins, not
absolute frequencies. `Frame::periods` contains periods measured in samples.

## Configuration and realtime use

`Config::default()` uses Quick with 24 bins per octave across five octaves.
`Config::desktop()` uses 64 bins per octave across six octaves. Progressive32
requires 24 bins per octave and five octaves:

```rust
use colorchord_rs::{Config, DftAlgorithm, NoteFinder};

let finder = NoteFinder::new(Config {
    algorithm: DftAlgorithm::Progressive32,
    ..Config::default()
});
```

Settings are immutable after construction. `Config::validate()` checks them
without allocating; construction validates automatically. `ConfigError`
exposes `field()` and `reason()`. Supported sample rates are 8–384 kHz, with
the highest configured bin below Nyquist. Bin counts must be even and 12–128;
octave counts are 1–8. See the API documentation for all limits.

Construct the finder outside the audio callback. Quick precomputes coefficients
and history indices, using about 526 KiB with defaults or 5.21 MiB with desktop
settings. An 8 MiB row-storage cap falls back to direct computation for larger
tables. Ingestion, analysis, and reset reuse storage without allocation, locks,
logging, or I/O. Reset preserves the tables and CPU backend.

Quick wraps its analysis windows within its configured history and uses `rtrb`
for batched history writes. Progressive32 keeps streaming transform state and
holds the newest sample until the next input. Each finder owns its mutable
state. Device capture and cross-thread transport are outside the library.

## License

MIT. Algorithms derive from ColorChord by Charles Lohr (CNLohr) and contributors.
ColorChord is Copyright 2015 Charles Lohr, Under the MIT/x11 License.
The attribution and license notices are retained in [LICENSE](LICENSE) and
[LICENSE-COLORCHORD](LICENSE-COLORCHORD).
