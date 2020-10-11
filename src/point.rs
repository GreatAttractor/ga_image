//
// ga_image - Image handling library
// Copyright (c) 2020 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//

//!
//! Basic geometry structs.
//!

use std::ops::{Add, AddAssign, Sub, Div};

pub struct PointFlt {
    pub x: f32,
    pub y: f32
}


impl std::fmt::Display for PointFlt {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "({:0.1}, {:0.1})", self.x, self.y)
    }
}


#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Point {
    pub x: i32,
    pub y: i32
}


impl Add for Point {
    type Output = Point;

    fn add(self, other: Point) -> Point {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}


impl Sub for Point {
    type Output = Point;

    fn sub(self, other: Point) -> Point {
        Point {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}


impl AddAssign for Point {
    fn add_assign(&mut self, other: Point) {
        *self = Point {
            x: self.x + other.x,
            y: self.y + other.y,
        };
    }
}

impl Div<i32> for Point {
    type Output = Point;

    fn div(self, d: i32) -> Point {
        Point{ x: self.x / d, y: self.y / d }
    }
}

impl std::fmt::Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "({}, {})", self.x, self.y)
    }
}


impl Point {
    pub fn sqr_dist(p1: &Point, p2: &Point) -> i32 {
        (p1.x - p2.x).pow(2) + (p1.y - p2.y).pow(2)
    }

    pub fn zero() -> Point { Point{ x: 0, y: 0 } }

    pub fn dist_from_origin(&self) -> f64 { ((self.x.pow(2) + self.y.pow(2)) as f64).sqrt() }

    pub fn vec_len(&self) -> f64 { self.dist_from_origin() }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32
}

impl Rect {
    pub fn contains_point(&self, p: &Point) -> bool {
        p.x >= self.x && p.x < self.x + self.width as i32 && p.y >= self.y && p.y < self.y + self.height as i32
    }


    pub fn contains_rect(&self, other: &Rect) -> bool {
        self.contains_point(&Point{ x: other.x, y: other.y }) &&
        self.contains_point(&Point{ x: other.x + other.width as i32 - 1,
                                    y: other.y + other.height as i32 - 1 })
    }


    pub fn pos(&self) -> Point { Point{ x: self.x, y: self.y } }

    pub fn set_pos(&mut self, pos: Point) { self.x = pos.x; self.y = pos.y; }
}
