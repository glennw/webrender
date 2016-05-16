/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::{Matrix4D, Point2D, Rect, Size2D};
use num_traits::Zero;
use time::precise_time_ns;

#[allow(dead_code)]
pub struct ProfileScope {
    name: &'static str,
    t0: u64,
}

impl ProfileScope {
    #[allow(dead_code)]
    pub fn new(name: &'static str) -> ProfileScope {
        ProfileScope {
            name: name,
            t0: precise_time_ns(),
        }
    }
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
        if self.name.chars().next() != Some(' ') {
            let t1 = precise_time_ns();
            let ms = (t1 - self.t0) as f64 / 1000000f64;
            println!("{} {}", self.name, ms);
        }
    }
}

// TODO: Implement these in euclid!
pub trait MatrixHelpers {
    fn transform_rect(&self, rect: &Rect<f32>) -> Rect<f32>;
}

impl MatrixHelpers for Matrix4D<f32> {
    #[inline]
    fn transform_rect(&self, rect: &Rect<f32>) -> Rect<f32> {
        let top_left = self.transform_point(&rect.origin);
        let top_right = self.transform_point(&rect.top_right());
        let bottom_left = self.transform_point(&rect.bottom_left());
        let bottom_right = self.transform_point(&rect.bottom_right());
        let (mut min_x, mut min_y) = (top_left.x.clone(), top_left.y.clone());
        let (mut max_x, mut max_y) = (min_x.clone(), min_y.clone());
        for point in &[ top_right, bottom_left, bottom_right ] {
            if point.x < min_x {
                min_x = point.x.clone()
            }
            if point.x > max_x {
                max_x = point.x.clone()
            }
            if point.y < min_y {
                min_y = point.y.clone()
            }
            if point.y > max_y {
                max_y = point.y.clone()
            }
        }
        Rect::new(Point2D::new(min_x.clone(), min_y.clone()),
                  Size2D::new(max_x - min_x, max_y - min_y))
    }
}

pub trait RectHelpers {
    fn from_points(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect<f32>;
    fn contains_rect(&self, other: &Rect<i32>) -> bool;
    //fn subtract(&self, other: &Rect<i32>) -> Vec<Rect<i32>>;
    fn is_well_formed_and_nonempty(&self) -> bool;
}

impl RectHelpers for Rect<i32> {
    fn contains_rect(&self, other: &Rect<i32>) -> bool {
        self.origin.x <= other.origin.x &&
        self.origin.y <= other.origin.y &&
        self.max_x() >= other.max_x() &&
        self.max_y() >= other.max_y()
    }

    fn from_points(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect<f32> {
        Rect::new(Point2D::new(x0, y0),
                  Size2D::new(x1 - x0, y1 - y0))
    }

    fn is_well_formed_and_nonempty(&self) -> bool {
        self.size.width > 0 && self.size.height > 0
    }

/*
    fn subtract(&self, other: &Rect<i32>) -> Vec<Rect<i32>> {
        if !self.intersects(other) {
            return vec![*self];
        }
        let mut result = Vec::new();

        let this_x0 = self.origin.x;
        let this_y0 = self.origin.y;
        let this_x1 = self.origin.x + self.size.width;
        let this_y1 = self.origin.y + self.size.height;

        let other_x0 = other.origin.x;
        let other_y0 = other.origin.y;
        let other_x1 = other.origin.x + other.size.width;
        let other_y1 = other.origin.y + other.size.height;

        let r = Rect::from_points(this_x0, this_y0, other_x0, this_y1);
        if r.is_well_formed_and_nonempty() {
            if let Some(r) = self.intersection(&r) {
                result.push(r);
            }
        }
        let r = Rect::from_points(other_x0, this_y0, other_x1, other_y0);
        if r.is_well_formed_and_nonempty() {
            if let Some(r) = self.intersection(&r) {
                result.push(r);
            }
        }
        let r = Rect::from_points(other_x0, other_y1, other_x1, this_y1);
        if r.is_well_formed_and_nonempty() {
            if let Some(r) = self.intersection(&r) {
                result.push(r);
            }
        }
        let r = Rect::from_points(other_x1, this_y0, this_x1, this_y1);
        if r.is_well_formed_and_nonempty() {
            if let Some(r) = self.intersection(&r) {
                result.push(r);
            }
        }

        //println!("SUBTRACT {:?} - {:?} = {:?}", self, other, result);

        result
    }
    */
}

// Don't use `euclid`'s `is_empty` because that has effectively has an "and" in the conditional
// below instead of an "or".
pub fn rect_is_empty<N:PartialEq + Zero>(rect: &Rect<N>) -> bool {
    rect.size.width == Zero::zero() || rect.size.height == Zero::zero()
}

pub fn rect_contains_rect(rect: &Rect<f32>, other: &Rect<f32>) -> bool {
    rect.origin.x <= other.origin.x &&
    rect.origin.y <= other.origin.y &&
    rect.max_x() >= other.max_x() &&
    rect.max_y() >= other.max_y()
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    (b - a) * t + a
}
