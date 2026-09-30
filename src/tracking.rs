// Derived from ColorChord's notefinder, filter, and turbo decomposition.
// Copyright 2015 Charles Lohr; see LICENSE-COLORCHORD.
use crate::{Config, Distribution, Note};

pub(crate) fn distance(a: f32, b: f32, modulus: f32) -> f32 {
    let d = (a - b).abs() % modulus;
    if d as f64 > modulus as f64 / 2.0 {
        modulus - d
    } else {
        d
    }
}
fn average(mut a: f32, aw: f32, mut b: f32, bw: f32, modulus: f32) -> f32 {
    if (a - b).abs() as f64 > modulus as f64 / 2.0 {
        if a < b {
            a += modulus;
        } else {
            b += modulus;
        }
    }
    ((a * aw + b * bw) / (aw + bw)) % modulus
}

pub(crate) fn spectrum(
    c: &Config,
    raw: &[f32],
    out: &mut [f32],
    folded: &mut [f32],
    scratch: &mut [f32],
) {
    let width = c.bins_per_octave;
    for (i, (out, raw)) in out.iter_mut().zip(raw).enumerate() {
        *out = ((*out * c.dft_iir) as f64
            + *raw as f64
                * (1.0 - c.dft_iir as f64)
                * c.amplification as f64
                * (1.0 + (c.slope * i as f32) as f64)) as f32;
    }
    for (i, bin) in out.iter_mut().take(width).enumerate() {
        *bin = (*bin as f64 * ((i as f64 + 1.0) / width as f64)) as f32;
    }
    for (i, bin) in out.iter_mut().rev().take(width).enumerate() {
        *bin = (*bin as f64 * ((i as f64 + 1.0) / width as f64)) as f32;
    }
    for (i, bin) in folded.iter_mut().enumerate() {
        *bin = 0.0;
        for octave in 0..c.octaves {
            *bin += out[i + octave * width];
        }
    }
    for _ in 0..c.filter_iterations {
        scratch.copy_from_slice(folded);
        for i in 0..width {
            let left = scratch[(i + width - 1) % width];
            let right = scratch[(i + 1) % width];
            folded[i] = (folded[i] as f64 * (1.0 - c.filter_strength as f64)
                + ((left + right) * c.filter_strength) as f64 * 0.5) as f32;
        }
    }
}

pub(crate) fn decompose(c: &Config, folded: &[f32], distributions: &mut [Distribution]) -> usize {
    let width = folded.len();
    let mut count = 0;
    for i in 0..width {
        let prev = folded[(i + width - 1) % width];
        let current = folded[i];
        let next = folded[(i + 1) % width];
        if prev >= current || next > current {
            continue;
        }
        let difference = (current - prev) + (current - next);
        let p = (current - prev) / difference;
        let n = (current - next) / difference;
        let offset = if p < n {
            -(0.5_f64 - p as f64)
        } else {
            0.5_f64 - n as f64
        } as f32;
        distributions[count] = Distribution {
            mean: i as f32 + offset,
            amplitude: current * 4.0,
            sigma: c.default_sigma,
            taken: false,
        };
        count += 1;
    }
    for dist in &mut distributions[count..] {
        *dist = Distribution {
            mean: -1.0,
            amplitude: 0.0,
            sigma: c.default_sigma,
            taken: false,
        };
    }
    if count > 0 {
        let total = distributions[..count]
            .iter()
            .fold(0.0_f32, |sum, d| sum + d.amplitude);
        let multiplier = c.compression_coefficient
            / (total * c.compression_coefficient).powf(c.compression_exponent);
        if multiplier.is_finite() {
            for d in &mut distributions[..count] {
                d.amplitude *= multiplier;
            }
        } else {
            // Tiny, valid input gains can underflow the f32 denominator or
            // overflow its reciprocal. Preserve f32 rounding on the normal
            // path, but normalize these edge cases before narrowing to f32.
            let coefficient = f64::from(c.compression_coefficient);
            let denominator =
                (f64::from(total) * coefficient).powf(f64::from(c.compression_exponent));
            for d in &mut distributions[..count] {
                d.amplitude = (f64::from(d.amplitude) * coefficient / denominator) as f32;
            }
        }
        // Stable insertion sort is bounded by the small number of peaks and never allocates.
        // Retain input order for equal amplitudes to keep tracking deterministic.
        for i in 1..count {
            let mut j = i;
            while j > 0 && distributions[j - 1].amplitude > distributions[j].amplitude {
                distributions.swap(j - 1, j);
                j -= 1;
            }
        }
    }
    count
}

pub(crate) fn track(c: &Config, dists: &mut [Distribution], notes: &mut [Note], next_id: &mut u32) {
    let width = c.bins_per_octave as f32;
    for note in notes.iter_mut() {
        note.matched = false;
        if let Some((index, dist)) = dists.iter_mut().enumerate().find(|(_, dist)| {
            !dist.taken
                && f64::from(dist.amplitude) > 0.00001
                && distance(note.position, dist.mean, width) < c.note_jumpability
        }) {
            note.distribution_index = index;
            dist.taken = true;
            if note.id == 0 {
                note.id = take_id(next_id);
            }
            note.matched = true;
            note.position = average(
                note.position,
                (1.0 - f64::from(c.note_attach_frequency_iir)) as f32,
                dist.mean,
                c.note_attach_frequency_iir,
                width,
            );
            note.amplitude = (f64::from(note.amplitude)
                * (1.0 - f64::from(c.note_attach_amplitude_iir))
                + f64::from(dist.amplitude * c.note_attach_amplitude_iir))
                as f32;
        }
    }

    for i in 0..notes.len() {
        for j in 0..notes.len() {
            if i == j {
                continue;
            }
            if notes[i].amplitude > 0.0
                && notes[j].amplitude > 0.0
                && distance(notes[i].position, notes[j].position, width) < c.note_combine_distance
            {
                let (a, b) = if notes[i].amplitude > notes[j].amplitude {
                    (i, j)
                } else {
                    (j, i)
                };
                let position = average(
                    notes[a].position,
                    notes[a].amplitude,
                    notes[b].position,
                    notes[b].amplitude,
                    width,
                );
                notes[a].amplitude += notes[b].amplitude;
                notes[a].position = position;
                notes[b].amplitude = 0.0;
                notes[b].position = -100.0;
                notes[b].id = 0;
            }
        }
    }
    for note in notes.iter_mut() {
        if note.amplitude < c.note_min_amplitude {
            note.id = 0;
            // Upstream intentionally visits every eligible peak here, without a break.
            for dist in dists.iter_mut() {
                if !dist.taken && dist.amplitude > c.note_minimum_new_distribution {
                    note.id = take_id(next_id);
                    dist.taken = true;
                    note.amplitude = dist.amplitude;
                    note.position = dist.mean;
                    note.matched = true;
                }
            }
        }
    }
    for note in notes.iter_mut() {
        if !note.matched {
            note.amplitude =
                (note.amplitude as f64 * (1.0 - c.note_attach_amplitude_iir as f64)) as f32;
        }
        note.amplitude_iir2 =
            (note.amplitude_iir2 as f64 * (1.0 - c.note_attach_amplitude_iir2 as f64)
                + (note.amplitude * c.note_attach_amplitude_iir2) as f64) as f32;
        if note.amplitude_iir2 < c.note_min_amplitude {
            note.amplitude = 0.0;
            note.amplitude_iir2 = 0.0;
        }
        note.amplitude_out = (note.amplitude - c.note_out_chop).max(0.0);
    }
}
fn take_id(next: &mut u32) -> u32 {
    let id = *next;
    *next = next.wrapping_add(1).max(1);
    id
}
