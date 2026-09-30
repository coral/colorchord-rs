use colorchord_rs::{Config, DftAlgorithm, NoteFinder};

fn tone(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 44100.0).sin() * 0.4)
        .collect()
}
#[test]
fn chunk_boundaries_instances_and_reset() {
    for algorithm in [DftAlgorithm::Quick, DftAlgorithm::Progressive32] {
        let config = Config {
            algorithm,
            ..Config::default()
        };
        let mut a = NoteFinder::new(config.clone()).unwrap();
        let mut b = NoteFinder::new(config.clone()).unwrap();
        let samples = tone(735 * 20);
        for block in samples.chunks(735) {
            a.push_samples(block);
            for tiny in block.chunks(13) {
                b.push_samples(tiny);
            }
            let af = a.analyze();
            let bf = b.analyze();
            assert_eq!(af.raw_bins, bf.raw_bins);
            assert_eq!(af.notes, bf.notes);
        }
        assert!(a.frame().notes.iter().any(|n| n.active()));
        a.reset();
        let mut fresh = NoteFinder::new(config).unwrap();
        for block in samples.chunks(735) {
            a.push_samples(block);
            fresh.push_samples(block);
            let af = a.analyze();
            let ff = fresh.analyze();
            assert_eq!(af.raw_bins, ff.raw_bins);
            assert_eq!(af.notes, ff.notes);
        }
    }
}
#[test]
fn silence_and_input_conversion() {
    for algorithm in [DftAlgorithm::Quick, DftAlgorithm::Progressive32] {
        let config = Config {
            algorithm,
            ..Config::default()
        };
        let mut finder = NoteFinder::new(config.clone()).unwrap();
        finder.push_samples(&[128_u8; 735]);
        let frame = finder.analyze();
        assert!(frame.raw_bins.iter().all(|&v| v == 0.0));
        assert!(frame.distributions.is_empty());
        assert!(frame.notes.iter().all(|n| !n.active()));
        let pcm: Vec<i16> = (0..735).map(|i| ((i % 31) * 1000 - 15000) as i16).collect();
        let floats: Vec<f32> = pcm.iter().map(|&v| v as f32 / 32768.0).collect();
        let mut other = NoteFinder::new(config).unwrap();
        finder.reset();
        finder.push_samples(&pcm);
        other.push_samples(&floats);
        assert_eq!(finder.analyze().raw_bins, other.analyze().raw_bins);
        finder.reset();
        finder.push_samples(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
        assert!(finder.analyze().raw_bins.iter().all(|&v| v == 0.0));
    }
}
#[test]
fn invalid_configuration_is_rejected() {
    for config in [
        Config {
            sample_rate: 0,
            ..Config::default()
        },
        Config {
            base_hz: f32::NAN,
            ..Config::default()
        },
        Config {
            bins_per_octave: 13,
            ..Config::default()
        },
        Config {
            algorithm: DftAlgorithm::Progressive32,
            octaves: 6,
            ..Config::default()
        },
        Config {
            history_size: 0,
            ..Config::default()
        },
        Config {
            note_attach_frequency_iir: 2.0,
            ..Config::default()
        },
    ] {
        assert!(NoteFinder::new(config).is_err());
    }
}
#[test]
fn valid_configuration_edges_are_finite() {
    for config in [
        Config {
            bins_per_octave: 12,
            octaves: 1,
            filter_iterations: 0,
            ..Config::default()
        },
        Config {
            bins_per_octave: 128,
            octaves: 8,
            ..Config::default()
        },
        Config::desktop(),
    ] {
        let mut finder = NoteFinder::new(config).unwrap();
        finder.push_samples(&tone(735));
        let frame = finder.analyze();
        assert!(frame.raw_bins.iter().all(|v| v.is_finite()));
        assert!(frame.folded.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn subnormal_signal_gain_does_not_produce_infinite_notes() {
    for (amplification, compression_coefficient) in
        [(1e-40, 1.0), (1e-35, 1e-6), (f32::from_bits(1), 1.0)]
    {
        let mut finder = NoteFinder::new(Config {
            amplification,
            compression_coefficient,
            compression_exponent: 1.0,
            ..Config::default()
        })
        .unwrap();
        for _ in 0..4 {
            finder.push_samples(&tone(735));
            let frame = finder.analyze();
            assert!(frame.distributions.iter().all(|d| d.amplitude.is_finite()));
            assert!(
                frame
                    .notes
                    .iter()
                    .all(|n| n.position.is_finite() && n.amplitude.is_finite())
            );
        }
    }
}

#[test]
fn configuration_errors_identify_the_field_and_constraint() {
    let config = Config {
        dft_iir: f32::NAN,
        ..Config::default()
    };
    let error = config.validate().unwrap_err();
    assert_eq!(error.field(), "dft_iir");
    assert!(error.reason().contains("0..=1"));
    assert!(error.to_string().contains("dft_iir"));
}

#[test]
fn long_chunks_and_empty_input_keep_transform_state_consistent() {
    for algorithm in [DftAlgorithm::Quick, DftAlgorithm::Progressive32] {
        let config = Config {
            algorithm,
            ..Config::default()
        };
        let mut whole = NoteFinder::new(config.clone()).unwrap();
        let mut chunks = NoteFinder::new(config).unwrap();
        let samples = tone(8096 * 3 + 19);
        whole.push_samples(&samples);
        for chunk in samples.chunks(37) {
            chunks.push_samples(chunk);
        }
        chunks.push_samples::<f32>(&[]);
        assert_eq!(whole.analyze().raw_bins, chunks.analyze().raw_bins);
        assert_eq!(whole.frame().notes, chunks.frame().notes);
        // An analysis without new input advances tracking, but not either DFT.
        let previous = whole.frame().raw_bins.to_vec();
        assert_eq!(whole.analyze().raw_bins, previous);
        whole.reset();
        assert!(whole.analyze().raw_bins.iter().all(|&v| v == 0.0));
    }
}

#[test]
fn color_helpers_handle_non_finite_input() {
    use colorchord_rs::color::{hsv_to_rgb, hsv_to_rgba, note_to_rgb, note_to_rgba};
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(note_to_rgb(value, 1.0, 1.0), [0.0; 3]);
        assert_eq!(hsv_to_rgb(0.0, value, 1.0), [0.0; 3]);
        assert_eq!(hsv_to_rgb(0.0, 1.0, value), [0.0; 3]);
        assert_eq!(note_to_rgba(value, 1.0, 1.0), 0xff);
        assert_eq!(hsv_to_rgba(value, 1.0, 1.0), 0xff);
    }
}
