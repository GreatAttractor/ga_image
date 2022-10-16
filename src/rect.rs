//
// ga_image - Image handling library
// Copyright (c) 2020-2022 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//

//!
//! Basic geometry structs.
//!

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32
}

pub type Point = [i32; 2];

pub const X: usize = 0;
pub const Y: usize = 1;

impl Rect {
    pub fn contains_point(&self, p: Point) -> bool {
        p[X] >= self.x && p[X] < self.x + self.width as i32 && p[Y] >= self.y && p[Y] < self.y + self.height as i32
    }

    pub fn contains_rect(&self, other: &Rect) -> bool {
        self.contains_point([other.x, other.y]) &&
        self.contains_point([other.x + other.width as i32 - 1,
                             other.y + other.height as i32 - 1])
    }

    pub fn pos(&self) -> Point { [self.x, self.y] }

    pub fn set_pos(&mut self, pos: Point) {
        self.x = pos[X];
        self.y = pos[Y];
    }

    pub fn size(&self) -> [u32; 2] { [self.width, self.height] }

    pub fn set_size(&mut self, size: [u32; 2]) {
        self.width = size[0];
        self.height = size[1];
    }

    pub fn intersection(&self, other: &Rect) -> Option<Rect> {
        let pos = if self.contains_point(other.pos()) {
            other.pos()
        } else if other.contains_point(self.pos()) {
            self.pos()
        } else {
            return None;
        };

        let max_x = (self.x + self.width as i32).min(other.x + other.width as i32);
        let max_y = (self.y + self.height as i32).min(other.y + other.height as i32);

        let width = (max_x - pos[X]) as u32;
        let height = (max_y - pos[Y]) as u32;

        Some(Rect{ x: pos[X], y: pos[Y], width, height })
    }
}
