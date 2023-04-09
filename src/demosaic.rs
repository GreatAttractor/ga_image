//
// ga_image - Image handling library
// Copyright (c) 2020 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//

//!
//! Demosaicing of raw color images.
//!

use crate::{CFAPattern, Image};
use num_traits::{AsPrimitive, PrimInt};

//TODO: after verification, use unchecked accesses

// offsets of components in a RGB triple
const RED: usize = 0;
const GREEN: usize = 1;
const BLUE: usize = 2;

fn get_two_lines_mut<T>(
    pixels: &mut [T],
    y1: usize,
    y2: usize,
    line_len: usize,
    stride: usize
) -> (&mut [T], &mut [T]) {
    assert!(y1 != y2);
    let ofs1 = y1 * stride;
    assert!(ofs1 + line_len <= pixels.len());

    let ofs2 = y2 * stride;
    assert!(ofs2 + line_len <= pixels.len());

    assert!(ofs1 + line_len <= ofs2 || ofs2 + line_len <= ofs1);

    (unsafe { std::slice::from_raw_parts_mut((&mut pixels[ofs1..]).as_mut_ptr(), line_len) },
     unsafe { std::slice::from_raw_parts_mut((&mut pixels[ofs2..]).as_mut_ptr(), line_len) })
}

/// Performs demosaicing of a 2x2 block.
///
/// `T2` (result type of RGB values) should be big enough to fix the sum of 3 `T1` (source) values.
///
/// # Parameters
///
///
/// * `dx_r` - X offset of the red input pixel in a 2x2 block.
/// * `dx_b` - X offset of the blue input pixel in a 2x2 block.
/// * `x` - X position of the block.
/// * `src_r` - Source "red" row.
/// * `src_r_m1` - Source row at y - 1 of source "red" row.
/// * `src_r_p1` - Source row at y + 1 of source "red" row.
/// * `src_b` - Source "blue" row.
/// * `src_b_m1` - Source row at y - 1 of source "blue" row.
/// * `src_b_p1` - Source row at y + 1 of source "blue" row.
/// * `rgb_at_r` - Output RGB values at red input pixel location.
/// * `rgb_at_b` - Output RGB values at blue input pixel location.
/// * `rgb_at_g_at_r_row` - Output RGB values at green input pixel at red row.
/// * `rgb_at_g_at_b_row` - Output RGB values at green input pixel at blue row.
///
fn demosaic_block_simple<T1: AsPrimitive<T2> + PrimInt, T2: PrimInt + 'static>(
    dx_r: usize,
    dx_b: usize,
    x: usize,
    src_r: &[T1],
    src_r_m1: &[T1],
    src_r_p1: &[T1],
    src_b: &[T1],
    src_b_m1: &[T1],
    src_b_p1: &[T1],
    rgb_at_r: &mut[T2; 3],
    rgb_at_b: &mut[T2; 3],
    rgb_at_g_at_r_row: &mut[T2; 3],
    rgb_at_g_at_b_row: &mut[T2; 3]
) {
    rgb_at_r[RED]   = src_r[x + dx_r].as_();
    rgb_at_r[GREEN] = (src_r[x + dx_r - 1].as_()
                     + src_r[x + dx_r + 1].as_()
                     + src_r_m1[x + dx_r].as_()
                     + src_r_p1[x + dx_r].as_()) >> 2;
    rgb_at_r[BLUE]  = (src_r_m1[x + dx_r - 1].as_()
                     + src_r_p1[x + dx_r + 1].as_()
                     + src_r_p1[x + dx_r - 1].as_()
                     + src_r_m1[x + dx_r + 1].as_()) >> 2;

    rgb_at_b[RED] = (src_b_m1[x + dx_b - 1].as_()
                   + src_b_p1[x + dx_b + 1].as_()
                   + src_b_p1[x + dx_b - 1].as_()
                   + src_b_m1[x + dx_b + 1].as_()) >> 2;

    rgb_at_b[GREEN] = (src_b[x + dx_b - 1].as_()
                     + src_b[x + dx_b + 1].as_()
                     + src_b_m1[x + dx_b].as_()
                     + src_b_p1[x + dx_b].as_()) >> 2;
    rgb_at_b[BLUE] = src_b[x + dx_b].as_();

    rgb_at_g_at_r_row[RED] = (src_r[x + (dx_r ^ 1) - 1].as_()
                            + src_r[x + (dx_r ^ 1) + 1].as_()) >> 1;
    rgb_at_g_at_r_row[GREEN] = src_r[x + dx_r ^ 1].as_();
    rgb_at_g_at_r_row[BLUE] = (src_r_m1[x + dx_r ^ 1].as_()
                             + src_r_p1[x + dx_r ^ 1].as_()) >> 1;

    rgb_at_g_at_b_row[RED] = (src_b_m1[x + dx_b ^ 1].as_()
                            + src_b_p1[x + dx_b ^ 1].as_()) >> 1;
    rgb_at_g_at_b_row[GREEN] = src_b[x + dx_b ^ 1].as_();
    rgb_at_g_at_b_row[BLUE] = (src_b[x + (dx_b ^ 1) - 1].as_()
                             + src_b[x + (dx_b ^ 1) + 1].as_()) >> 1;
}

/// The implementing type is that of the weighted sum of raw CFA values; i16 for u8 inputs, i32 for u16 inputs.
/// `Output` is the type of output RGB values.
trait ConvertToOutput<Output: PrimInt> {
    fn to_output(&self) -> Output;
}

impl ConvertToOutput<u8> for i16 {
    fn to_output(&self) -> u8 { *self as u8 }
}

impl ConvertToOutput<u8> for i32 {
    fn to_output(&self) -> u8 { (self >> 8) as u8 }
}

/// Performs demosaicing of a 2x2 block.
///
/// `Sum` (result type of RGB values) must be bigger than `Source`, as it needs to hold many multiples of source values.
///
/// TODO: describe parameters
///
fn demosaic<Source, Sum, Output>(
    width: usize,
    height: usize,
    input: &[Source],
    input_stride: usize,
    output: &mut [Output],
    output_stride: usize,
    cfa_pattern: CFAPattern
) where
    Source: AsPrimitive<Sum> + PrimInt,
    Sum: PrimInt + 'static + ConvertToOutput<Output>,
    Output: PrimInt
{
    if width < 6 || height < 6 { return; }

    // offset of the red input pixel in each 2x2 block
    let dx_r = cfa_pattern.red_col_ofs() as usize;
    let dy_r = cfa_pattern.red_row_ofs() as usize;

    // offset of the blue input pixel in each 2x2 block
    let dx_b = dx_r as usize ^ 1;
    let dy_b = dy_r as usize ^ 1;

    // demosaic a 2x2 block at a time
    for y in (2..height - 2).step_by(2) {
        // current source "red" row and its neighbors
        let src_r_m1 = &input[(y + dy_r - 1) * input_stride..];
        let src_r    = &input[(y + dy_r)     * input_stride..];
        let src_r_p1 = &input[(y + dy_r + 1) * input_stride..];

        // current source "blue" row and its neighbors
        let src_b_m1 = &input[(y + dy_b - 1) * input_stride..];
        let src_b    = &input[(y + dy_b)     * input_stride..];
        let src_b_p1 = &input[(y + dy_b + 1) * input_stride..];

        // current destination "red" and "blue" row
        let (dest_r, dest_b) =
            get_two_lines_mut(output, y + dy_r, y + dy_b, width * 3, output_stride);

        for x in (2..width - 2).step_by(2) {
            let mut rgb_at_r = [Sum::zero(); 3]; // RGB values at red input pixel location
            let mut rgb_at_b = [Sum::zero(); 3]; // RGB values at blue input pixel location
            let mut rgb_at_g_at_r_row = [Sum::zero(); 3]; // RGB values at green input pixel at red row
            let mut rgb_at_g_at_b_row = [Sum::zero(); 3]; // RGB values at green input pixel at blue row

            demosaic_block_simple(
                dx_r, dx_b,
                x,
                src_r, src_r_m1, src_r_p1,
                src_b, src_b_m1, src_b_p1,
                &mut rgb_at_r,
                &mut rgb_at_b,
                &mut rgb_at_g_at_r_row,
                &mut rgb_at_g_at_b_row
            );

            dest_r[(x + dx_r) * 3 + RED]   = rgb_at_r[RED].to_output();
            dest_r[(x + dx_r) * 3 + GREEN] = rgb_at_r[GREEN].to_output();
            dest_r[(x + dx_r) * 3 + BLUE]  = rgb_at_r[BLUE].to_output();

            dest_b[(x + dx_b) * 3 + RED]   = rgb_at_b[RED].to_output();
            dest_b[(x + dx_b) * 3 + GREEN] = rgb_at_b[GREEN].to_output();
            dest_b[(x + dx_b) * 3 + BLUE]  = rgb_at_b[BLUE].to_output();

            dest_r[(x + dx_b) * 3 + RED]   = rgb_at_g_at_r_row[RED].to_output();
            dest_r[(x + dx_b) * 3 + GREEN] = rgb_at_g_at_r_row[GREEN].to_output();
            dest_r[(x + dx_b) * 3 + BLUE]  = rgb_at_g_at_r_row[BLUE].to_output();

            dest_b[(x + dx_r) * 3 + RED]   = rgb_at_g_at_b_row[RED].to_output();
            dest_b[(x + dx_r) * 3 + GREEN] = rgb_at_g_at_b_row[GREEN].to_output();
            dest_b[(x + dx_r) * 3 + BLUE]  = rgb_at_g_at_b_row[BLUE].to_output();
        }
    }

    //TODO: fill borders
}

/// Performs demosaicing of 8-bit `input` and saves as 8-bit RGB in `output`.
pub fn demosaic_raw8_as_rgb8(
    width: usize,
    height: usize,
    input: &[u8],
    input_stride: usize,
    output: &mut [u8],
    output_stride: usize,
    cfa_pattern: CFAPattern
) {
    demosaic::<u8, i16, u8>(
        width,
        height,
        input,
        input_stride,
        output,
        output_stride,
        cfa_pattern
    );
}

/// Performs demosaicing of 16-bit `input` and saves as 8-bit RGB in `output`.
pub fn demosaic_raw16_as_rgb8(
    width: usize,
    height: usize,
    input: &[u16],
    input_stride: usize,
    output: &mut [u8],
    output_stride: usize,
    cfa_pattern: CFAPattern
) {
    demosaic::<u16, i32, u8>(
        width,
        height,
        input,
        input_stride,
        output,
        output_stride,
        cfa_pattern
    );
}

/// Performs demosaicing of 8-bit `src` and saves as 8-bit mono in `dest`.
pub fn demosaic_raw8_as_mono8(_src: &Image, _dest: &mut Image) {
    panic!("Not implemented yet.");
}

/// Performs demosaicing of 16-bit `src` and saves as 16-bit RGB in `dest`.
pub fn demosaic_raw16_as_rgb16(_src: &Image, _dest: &mut Image) {
    panic!("Not implemented yet.");
}

/// Performs demosaicing of 16-bit `src` and saves as 8-bit mono in `dest`.
pub fn demosaic_raw16_as_mono8(_src: &Image, _dest: &mut Image) {
    panic!("Not implemented yet.");
}
