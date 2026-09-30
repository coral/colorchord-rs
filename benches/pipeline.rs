//! Stage timings without file decoding, resampling, or report I/O in the timed region.
//! Run: cargo bench --bench pipeline -- [filter] > target/perf/results.json
//! BENCH_MS and BENCH_ROUNDS control measurement duration (defaults: 100 ms, 7 rounds).
#[allow(dead_code, unused_imports)] // Unit-test helpers are unused in this standalone bench.
#[path = "../src/dft.rs"]
mod dft;
#[path = "../src/tracking.rs"]
mod tracking;
pub use colorchord_rs::{Config, DftAlgorithm, Distribution, FromSample, Note, NoteFinder, Sample};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

fn measure(name: &str, mut op: impl FnMut(usize), results: &mut Vec<serde_json::Value>) {
    if std::env::args()
        .nth(1)
        .is_some_and(|f| f != "--bench" && !name.contains(&f))
    {
        return;
    }
    let duration = Duration::from_millis(
        std::env::var("BENCH_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100),
    );
    let rounds: usize = std::env::var("BENCH_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    assert!(rounds > 0 && !duration.is_zero());
    for i in 0..64 {
        op(i);
    }
    let mut samples = Vec::new();
    for _ in 0..rounds {
        let start = Instant::now();
        let mut iterations = 0;
        loop {
            for _ in 0..32 {
                op(iterations);
                iterations += 1;
            }
            if start.elapsed() >= duration {
                break;
            }
        }
        samples.push(start.elapsed().as_secs_f64() * 1e9 / iterations as f64);
    }
    samples.sort_by(f64::total_cmp);
    results.push(serde_json::json!({"name":name, "median_ns":samples[rounds / 2],
        "min_ns":samples[0],"max_ns":samples[rounds - 1],"rounds":rounds,"round_ms":duration.as_millis()}));
}
fn main() {
    let input: Vec<f32> = (0..735 * 64)
        .map(|i| {
            let t = i as f32 / 44100.0;
            0.2 * ((t * 220.0 * std::f32::consts::TAU).sin()
                + (t * 329.6276 * std::f32::consts::TAU).sin()
                + (t * 440.0 * std::f32::consts::TAU).sin())
        })
        .collect();
    let block = |i: usize| &input[(i % 64) * 735..(i % 64 + 1) * 735];
    let mut results = Vec::new();
    let mut tables = serde_json::Map::new();
    for (name, config) in [
        ("quick-defaults", Config::default()),
        ("quick-desktop", Config::desktop()),
        (
            "progressive32",
            Config {
                algorithm: DftAlgorithm::Progressive32,
                ..Config::default()
            },
        ),
    ] {
        measure(
            &format!("{name}/construct"),
            |_| {
                black_box(NoteFinder::new(black_box(config.clone())).unwrap());
            },
            &mut results,
        );
        let mut finder = NoteFinder::new(config.clone()).unwrap();
        measure(
            &format!("{name}/pipeline"),
            |i| {
                finder.push_samples(black_box(block(i)));
                black_box(finder.analyze());
            },
            &mut results,
        );
        measure(
            &format!("{name}/ingest"),
            |i| finder.push_samples(black_box(block(i))),
            &mut results,
        );
        let mut transform = dft::Transform::new(&config, finder.frame().periods);
        tables.insert(name.into(), transform.coefficient_table_bytes().into());
        transform.reset();
        transform.push_samples(&input);
        let bins = config.bins_per_octave * config.octaves;
        let mut raw = vec![0.0; bins];
        measure(
            &format!("{name}/dft-output"),
            |_| {
                transform.analyze(black_box(&mut raw));
                black_box(&raw);
            },
            &mut results,
        );
        #[cfg(feature = "simd")]
        for level in supported_levels() {
            transform.set_level(level);
            measure(
                &format!("{name}/dft-block/{level:?}"),
                |i| {
                    transform.push_samples(black_box(block(i)));
                    transform.analyze(black_box(&mut raw));
                    black_box(&raw);
                },
                &mut results,
            );
        }
        transform.reset();
        transform.push_samples(&input);
        transform.analyze(&mut raw);
        let mut out = vec![0.0; bins];
        let mut folded = vec![0.0; config.bins_per_octave];
        let mut scratch = folded.clone();
        measure(
            &format!("{name}/spectrum"),
            |_| {
                tracking::spectrum(
                    &config,
                    black_box(&raw),
                    &mut out,
                    &mut folded,
                    &mut scratch,
                );
                black_box(&folded);
            },
            &mut results,
        );
        tracking::spectrum(&config, &raw, &mut out, &mut folded, &mut scratch);
        let mut dists = vec![Distribution::default(); config.bins_per_octave / 2];
        let mut notes = vec![Note::default(); config.bins_per_octave / 2];
        let mut next_id = 1;
        measure(
            &format!("{name}/decompose-track"),
            |_| {
                let count = tracking::decompose(&config, black_box(&folded), &mut dists);
                tracking::track(&config, &mut dists[..count], &mut notes, &mut next_id);
                black_box(&notes);
            },
            &mut results,
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "architecture":std::env::consts::ARCH,"sample_rate":44100,"samples_per_frame":735,
            "simd":cfg!(feature = "simd"), "coefficient_table_bytes":tables,
            "results":results
        }))
        .unwrap()
    );
}

#[cfg(feature = "simd")]
fn supported_levels() -> Vec<fearless_simd::Level> {
    use fearless_simd::Level;
    let best = Level::new();
    #[allow(unused_mut)]
    let mut levels = vec![best];
    #[cfg(target_arch = "x86_64")]
    {
        levels.extend(best.as_sse2().map(Level::Sse2));
        levels.extend(best.as_sse4_2().map(Level::Sse4_2));
        levels.extend(best.as_avx2().map(Level::Avx2));
    }
    levels
}
