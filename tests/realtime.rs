use colorchord_rs::{Config, DftAlgorithm, NoteFinder};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}
struct Counting;
// SAFETY: all memory operations are forwarded unchanged to the system allocator.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        TRACK.with(|enabled| {
            if enabled.get() {
                ALLOCATIONS.with(|n| n.set(n.get() + 1));
            }
        });
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        TRACK.with(|enabled| {
            if enabled.get() {
                ALLOCATIONS.with(|n| n.set(n.get() + 1));
            }
        });
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;
#[test]
fn processing_and_reset_do_not_allocate() {
    for config in [
        Config::default(),
        Config::desktop(),
        Config {
            algorithm: DftAlgorithm::Progressive32,
            ..Config::default()
        },
    ] {
        let mut finder = NoteFinder::new(config).unwrap();
        let samples: Vec<f32> = (0..735).map(|i| (i as f32 * 0.062).sin() * 0.5).collect();
        ALLOCATIONS.with(|n| n.set(0));
        TRACK.with(|v| v.set(true));
        for _ in 0..30 {
            finder.push_samples(&samples);
            std::hint::black_box(finder.analyze());
        }
        finder.reset();
        TRACK.with(|v| v.set(false));
        assert_eq!(ALLOCATIONS.with(Cell::get), 0);
    }
}
