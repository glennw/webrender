/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::{Size2D, Point2D, Rect, Matrix4D};
use std::str::FromStr;
use app_units::Au;

use yaml_rust::Yaml;

use webrender_traits::*;

pub trait YamlHelper {
    fn as_force_f32(&self) -> Option<f32>;
    fn as_vec_f32(&self) -> Option<Vec<f32>>;
    fn as_vec_u32(&self) -> Option<Vec<u32>>;
    fn as_rect(&self) -> Option<Rect<f32>>;
    fn as_size(&self) -> Option<Size2D<f32>>;
    fn as_point(&self) -> Option<Point2D<f32>>;
    fn as_matrix4d(&self) -> Option<Matrix4D<f32>>;
    fn as_colorf(&self) -> Option<ColorF>;
    fn as_vec_colorf(&self) -> Option<Vec<ColorF>>;
    fn as_complex_clip_rect(&self) -> Option<ComplexClipRegion>;
    fn as_clip_region(&self, &mut DisplayListBuilder) -> Option<ClipRegion>;
    fn as_px_to_au(&self) -> Option<Au>;
    fn as_pt_to_au(&self) -> Option<Au>;
    fn as_vec_string(&self) -> Option<Vec<String>>;
    fn as_border_radius(&self) -> Option<BorderRadius>;
}

fn string_to_color(color: &str) -> Option<ColorF> {
    match color {
        "red" => Some(ColorF::new(1.0, 0.0, 0.0, 1.0)),
        "green" => Some(ColorF::new(0.0, 1.0, 0.0, 1.0)),
        "blue" => Some(ColorF::new(0.0, 0.0, 1.0, 1.0)),
        "white" => Some(ColorF::new(1.0, 1.0, 1.0, 1.0)),
        "black" => Some(ColorF::new(0.0, 0.0, 0.0, 1.0)),
        s => {
            let items: Vec<f32> = s.split_whitespace().map(|s| f32::from_str(s).unwrap()).collect();
            if items.len() == 3 {
                Some(ColorF::new(items[0] / 255.0, items[1] / 255.0, items[2] / 255.0, 1.0))
            } else if items.len() == 4 {
                Some(ColorF::new(items[0] / 255.0, items[1] / 255.0, items[2] / 255.0, items[3]))
            } else {
                None
            }
        }
    }
}

impl YamlHelper for Yaml {
    fn as_force_f32(&self) -> Option<f32> {
        match *self {
            Yaml::Integer(iv) => Some(iv as f32),
            Yaml::String(ref sv) | Yaml::Real(ref sv) => match f32::from_str(sv.as_str()) {
                Ok(v) => Some(v),
                Err(_) => None
            },
            _ => None
        }
    }

    fn as_vec_f32(&self) -> Option<Vec<f32>> {
        if let Some(v) = self.as_str() {
            Some(v.split_whitespace()
                 .map(|v| f32::from_str(v).expect(&format!("expected float value, got '{:?}'", v)))
                 .collect())
        } else if let Some(v) = self.as_vec() {
            Some(v.iter().map(|v| {
                match *v {
                    Yaml::Integer(k) => k as f32,
                    Yaml::String(ref k) | Yaml::Real(ref k) => f32::from_str(&k).expect(&format!("expected float value, got '{:?}'", v)),
                    _ => panic!("expected float value, got '{:?}'", v),
                }
            }).collect())
        } else {
            None
        }
    }

    fn as_vec_u32(&self) -> Option<Vec<u32>> {
        if let Some(v) = self.as_vec() {
            Some(v.iter().map(|v| v.as_i64().unwrap() as u32).collect())
        } else {
            None
        }
    }

    fn as_px_to_au(&self) -> Option<Au> {
        match self.as_force_f32() {
            Some(fv) => Some(Au::from_f32_px(fv)),
            None => None
        }
    }

    fn as_pt_to_au(&self) -> Option<Au> {
        match self.as_force_f32() {
            Some(fv) => Some(Au::from_f32_px(fv * 16. / 12.)),
            None => None
        }
    }

    fn as_rect(&self) -> Option<Rect<f32>> {
        if self.is_badvalue() {
            return None;
        }

        let nums = self.as_vec_f32().unwrap();
        if nums.len() != 4 {
            panic!("rect expected 4 float values, got {} instead ('{:?}')", nums.len(), self);
        }
        Some(Rect::new(Point2D::new(nums[0], nums[1]), Size2D::new(nums[2], nums[3])))
    }

    fn as_size(&self) -> Option<Size2D<f32>> {
        if self.is_badvalue() {
            return None;
        }

        let nums = self.as_vec_f32().unwrap();
        if nums.len() != 2 {
            panic!("size expected 2 float values, got {} instead ('{:?}')", nums.len(), self);
        }
        Some(Size2D::new(nums[0], nums[1]))
    }

    fn as_point(&self) -> Option<Point2D<f32>> {
        if self.is_badvalue() {
            return None;
        }

        let nums = self.as_vec_f32().unwrap();
        if nums.len() != 2 {
            panic!("point expected 2 float values, got {} instead ('{:?}')", nums.len(), self);
        }
        Some(Point2D::new(nums[0], nums[1]))
    }

    fn as_matrix4d(&self) -> Option<Matrix4D<f32>> {
        None
    }

    fn as_colorf(&self) -> Option<ColorF> {
        if let Some(mut nums) = self.as_vec_f32() {
            if nums.len() != 3 && nums.len() != 4 {
                panic!("color expected a color name, or 3-4 floats; got '{:?}'", self);
            }

            if nums.len() == 3 {
                nums.push(1.0);
            }
            return Some(ColorF::new(nums[0] / 255.0, nums[1] / 255.0, nums[2] / 255.0, nums[3]));
        }

        if let Some(s) = self.as_str() {
            string_to_color(s)
        } else {
            None
        }
    }

    fn as_vec_colorf(&self) -> Option<Vec<ColorF>> {
        if let Some(v) = self.as_vec() {
            Some(v.iter().map(|v| v.as_colorf().unwrap()).collect())
        } else {
            if let Some(color) = self.as_colorf() {
                Some(vec![color])
            } else {
                None
            }
        }
    }

    fn as_complex_clip_rect(&self) -> Option<ComplexClipRegion> {
        if self.is_badvalue() {
            return None;
        }

        let nums = self.as_vec_f32().unwrap();
        match nums.len() {
            4 => Some(ComplexClipRegion::new(Rect::new(Point2D::new(nums[0], nums[1]), Size2D::new(nums[2], nums[3])),
                                             BorderRadius::zero())),
            5 => Some(ComplexClipRegion::new(Rect::new(Point2D::new(nums[0], nums[1]), Size2D::new(nums[2], nums[3])),
                                             BorderRadius::uniform(nums[4]))),
            8 => Some(ComplexClipRegion::new(Rect::new(Point2D::new(nums[0], nums[1]), Size2D::new(nums[2], nums[3])),
                                             BorderRadius {
                                                 top_left: Size2D::new(nums[4], nums[4]),
                                                 top_right: Size2D::new(nums[5], nums[5]),
                                                 bottom_left: Size2D::new(nums[6], nums[6]),
                                                 bottom_right: Size2D::new(nums[7], nums[7]),
                                             })),
            12 => Some(ComplexClipRegion::new(Rect::new(Point2D::new(nums[0], nums[1]), Size2D::new(nums[2], nums[3])),
                                              BorderRadius {
                                                  top_left: Size2D::new(nums[4], nums[5]),
                                                  top_right: Size2D::new(nums[6], nums[7]),
                                                  bottom_left: Size2D::new(nums[8], nums[9]),
                                                  bottom_right: Size2D::new(nums[10], nums[11]),
                                              })),
            n => panic!("complex clip rect expected 4, 5, 8, or 12 floats; got {} instead at '{:?}'", n, self),
        }
    }

    fn as_clip_region(&self, builder: &mut DisplayListBuilder) -> Option<ClipRegion> {
        if self.is_badvalue() {
            return None;
        }

        // TODO add support for clip masks
        // TODO add support for rounded rect clips

        // if it's not a vec, then assume it's a single rect
        if self.as_vec().is_none() {
            let rect = self.as_rect().expect(&format!("clip region '{:?}', thought it was a rect but it's not?", self));
            return Some(builder.new_clip_region(&rect, Vec::new(), None));
        }

        // otherwise it's an array of complex clip rects
        let mut bounds = Rect::<f32>::zero();
        let mut clips = Vec::<ComplexClipRegion>::new();

        for item in self.as_vec().unwrap() {
            let c = item.as_complex_clip_rect().unwrap();
            bounds = bounds.union(&c.rect);
            clips.push(c);
        }

        Some(builder.new_clip_region(&bounds, clips, None))
    }

    fn as_vec_string(&self) -> Option<Vec<String>> {
        if let Some(v) = self.as_vec() {
            Some(v.iter().map(|v| v.as_str().unwrap().to_owned()).collect())
        } else if let Some(s) = self.as_str() {
            Some(vec![s.to_owned()])
        } else {
            None
        }
    }

    fn as_border_radius(&self) -> Option<BorderRadius> {
        let top_left = self["top_right"].as_size().unwrap_or(Size2D::zero());
        let top_right = self["top_right"].as_size().unwrap_or(Size2D::zero());
        let bottom_left = self["bottom_left"].as_size().unwrap_or(Size2D::zero());
        let bottom_right = self["bottom_right"].as_size().unwrap_or(Size2D::zero());
        Some(BorderRadius {
            top_left: top_left,
            top_right: top_right,
            bottom_left: bottom_left,
            bottom_right: bottom_right,
        })
    }
}
