//! Pitch-class and HSV color conversion.
//! Derived from Charles Lohr's public-domain color algorithms; see LICENSE-COLORCHORD.

/// Map a pitch class (one octave = 1.0) to RGB. Saturation/value are normally 0..=1.
/// Non-finite inputs produce black.
pub fn note_to_rgb(note: f32, saturation: f32, value: f32) -> [f32; 3] {
    hsv_to_rgb(note_hue(note), saturation, value)
}
fn note_hue(note: f32) -> f32 {
    let note = (note % 1.0) * 12.0;
    if note < 4.0 {
        ((4.0 - note as f64) / 24.0) as f32
    } else if note < 8.0 {
        ((4.0 - note as f64) / 12.0) as f32
    } else {
        ((12.0 - note as f64) / 8.0 + 1.0 / 6.0) as f32
    }
}
/// Convert HSV using the pitch visualization hue rotation and channel order.
/// Non-finite inputs produce black.
pub fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> [f32; 3] {
    if !hue.is_finite() || !saturation.is_finite() || !value.is_finite() {
        return [0.0; 3];
    }
    let rotation = ((hue as f64 * 6.0) % 6.0) as f32;
    let rotation = ((rotation as f64 + 6.0 + 1.0) % 6.0) as f32;
    let (mut r, mut g, mut b) = if rotation < 1.0 {
        (1.0, (1.0 - rotation as f64) as f32, 0.0)
    } else if rotation < 2.0 {
        (1.0, 0.0, (rotation as f64 - 1.0) as f32)
    } else if rotation < 3.0 {
        ((3.0 - rotation as f64) as f32, 0.0, 1.0)
    } else if rotation < 4.0 {
        (0.0, (rotation as f64 - 3.0) as f32, 1.0)
    } else if rotation < 5.0 {
        (0.0, 1.0, (5.0 - rotation as f64) as f32)
    } else {
        ((rotation as f64 - 5.0) as f32, 1.0, 0.0)
    };
    r *= value;
    g *= value;
    b *= value;
    let sum = r + g + b;
    let blend = |channel: f32| {
        ((channel * saturation) as f64 + sum as f64 * (1.0 - saturation as f64)) as f32
    };
    [
        blend(r).clamp(0.0, 1.0),
        blend(b).clamp(0.0, 1.0),
        blend(g).clamp(0.0, 1.0),
    ]
}
/// Pack pitch-class color as `0xRRGGBBAA`, truncating channel values to bytes.
pub fn note_to_rgba(note: f32, saturation: f32, value: f32) -> u32 {
    pack(note_to_rgb(note, saturation, value))
}
/// Pack HSV color as `0xRRGGBBAA`, truncating channel values to bytes.
pub fn hsv_to_rgba(hue: f32, saturation: f32, value: f32) -> u32 {
    pack(hsv_to_rgb(hue, saturation, value))
}
fn pack(rgb: [f32; 3]) -> u32 {
    u32::from_be_bytes([
        (rgb[0] * 255.0) as u8,
        (rgb[1] * 255.0) as u8,
        (rgb[2] * 255.0) as u8,
        255,
    ])
}
