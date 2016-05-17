/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use batch_builder::{BorderSideHelpers, BoxShadowMetrics};
use device::{TextureId, TextureFilter};
use euclid::{Point2D, Rect, Matrix4D, Size2D, Point4D};
use fnv::FnvHasher;
use frame::FrameId;
use internal_types::{AxisDirection, BoxShadowRasterOp, Glyph, GlyphKey, RasterItem, RectUv};
use quadtree::Quadtree;
use renderer::{BLUR_INFLATION_FACTOR, TEXT_TARGET_SIZE};
use resource_cache::ResourceCache;
use resource_list::ResourceList;
use std::cmp::{self, Ordering};
use std::collections::{HashMap, HashSet};
use std::f32;
use std::mem;
use std::hash::{Hash, BuildHasherDefault};
use texture_cache::{TexturePage};
use util::{self, RectHelpers, MatrixHelpers, subtract_rect};
use webrender_traits::{ColorF, FontKey, ImageKey, ImageRendering, ComplexClipRegion};
use webrender_traits::{BorderDisplayItem, BorderStyle, ItemRange, AuxiliaryLists, BorderRadius};
use webrender_traits::{BoxShadowClipMode, PipelineId};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum EventKind {
    Begin,
    End,
}

#[derive(Eq, PartialEq, Debug)]
struct Event {
    kind: EventKind,
    value: Au,
    key: usize,
}

impl Event {
    fn begin(value: f32, key: usize) -> Event {
        Event {
            kind: EventKind::Begin,
            value: Au::from_f32_px(value),
            key: key,
        }
    }

    fn end(value: f32, key: usize) -> Event {
        Event {
            kind: EventKind::End,
            value: Au::from_f32_px(value),
            key: key,
        }
    }
}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Event) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Event) -> Ordering {
        if self.value != other.value {
            return self.value.partial_cmp(&other.value).unwrap();
        }

        if self.key != other.key {
            return self.key.cmp(&other.key)
        }

        match (self.kind, other.kind) {
            (EventKind::Begin, EventKind::Begin) => Ordering::Equal,
            (EventKind::End, EventKind::End) => Ordering::Equal,
            (EventKind::Begin, EventKind::End) => Ordering::Greater,
            (EventKind::End, EventKind::Begin) => Ordering::Less,
        }
    }
}

struct OpenRect {
    keys: HashSet<usize>,
    x0: f32,
    y0: f32,
    y1: f32,
}

impl OpenRect {
    fn new(keys: HashSet<usize>, x0: f32, y0: f32, y1: f32) -> OpenRect {
        OpenRect {
            keys: keys,
            x0: x0,
            y0: y0,
            y1: y1,
        }
    }
}

struct Sap {
    //rectangles: Vec<Rect<f32>>,
    //keys: Vec<T>,
}

impl Sap {
    fn new() -> Sap {
        Sap {
            //rectangles: Vec::new(),
            //keys: Vec::new(),
        }
    }

/*
    #[inline]
    fn push(&mut self, rect: &Rect<f32>, key: T) {
        self.rectangles.push(*rect);
        self.keys.push(key);
    }*/

    fn doit<F>(&self,
               bounding_rect: &Rect<f32>,
               parts: &Vec<PrimitivePart>,
               mut f: F) where F: FnMut(&Rect<f32>, Vec<usize>) {

        let event_count = parts.len() * 2;
        let mut x_events = Vec::with_capacity(event_count);
        for (part_index, part) in parts.iter().enumerate() {
            if bounding_rect.intersects(&part.rect) {
                let px0 = part.rect.origin.x.max(bounding_rect.origin.x);
                let px1 = (part.rect.origin.x + part.rect.size.width).min(bounding_rect.origin.x + bounding_rect.size.width);
                debug_assert!(px0 != px1);

                x_events.push(Event::begin(px0, part_index));
                x_events.push(Event::end(px1, part_index));
            }
        }
        x_events.sort();

        let mut open_rects: Vec<OpenRect> = Vec::new();

        for xe in x_events {
            let rect = &parts[xe.key].rect;

            match xe.kind {
                EventKind::Begin => {
                    let py0 = rect.origin.y.max(bounding_rect.origin.y);
                    let py1 = (rect.origin.y + rect.size.height).min(bounding_rect.origin.y + bounding_rect.size.height);
                    debug_assert!(py0 != py1);

                    let mut y_events = Vec::new();
                    y_events.push(Event::begin(py0, xe.key));
                    y_events.push(Event::end(py1, xe.key));

                    let mut new_open_rects = Vec::new();
                    for open_rect in open_rects.drain(..) {
                        if open_rect.y0 < py1 && py0 < open_rect.y1 {
                            if xe.value.to_f32_px() > open_rect.x0 {
                                // TODO(gw): Fix extra key dereferencing here!
                                let mut user_keys = Vec::new();
                                for key in &open_rect.keys {
                                    user_keys.push(*key);
                                }
                                f(&Rect::from_points(open_rect.x0,
                                                     open_rect.y0,
                                                     xe.value.to_f32_px(),
                                                     open_rect.y1),
                                  user_keys);
                            }
                            for key in open_rect.keys {
                                y_events.push(Event::begin(open_rect.y0, key));
                                y_events.push(Event::end(open_rect.y1, key));
                            }
                        } else {
                            new_open_rects.push(open_rect);
                        }
                    }
                    open_rects = new_open_rects;

                    y_events.sort();
                    let mut active_y = HashSet::new();
                    let mut y0 = y_events[0].value;

                    for i in 0..y_events.len() {
                        let ye = &y_events[i];

                        if i == y_events.len()-1 || (i > 0 && ye.value > y_events[i-1].value) {
                            if active_y.len() > 0 && ye.value > y0 {
                                open_rects.push(OpenRect::new(active_y.clone(), xe.value.to_f32_px(), y0.to_f32_px(), ye.value.to_f32_px()));
                            }
                            y0 = ye.value;
                        }

                        match ye.kind {
                            EventKind::Begin => {
                                active_y.insert(ye.key);
                            }
                            EventKind::End => {
                                active_y.remove(&ye.key);
                            }
                        }
                    }
                }
                EventKind::End => {
                    let mut new_open_rects = Vec::new();
                    for mut rect in open_rects.drain(..) {
                        if rect.keys.contains(&xe.key) {
                            if xe.value.to_f32_px() > rect.x0 {
                                // TODO(gw): Fix extra key dereferencing here!
                                let mut user_keys = Vec::new();
                                for key in &rect.keys {
                                    user_keys.push(*key);
                                }
                                f(&Rect::from_points(rect.x0,
                                                     rect.y0,
                                                     xe.value.to_f32_px(),
                                                     rect.y1),
                                  user_keys);
                            }
                            rect.keys.remove(&xe.key);
                            rect.x0 = xe.value.to_f32_px();
                            if rect.keys.len() > 0 {
                                new_open_rects.push(rect);
                            }
                        } else {
                            new_open_rects.push(rect);
                        }
                    }
                    open_rects = new_open_rects;
                }
            }
        }
    }
}

fn sap<F>(bounding_rect: &Rect<f32>,
          parts: &Vec<PrimitivePart>,
          mut f: F) where F: FnMut(Rect<f32>, &HashSet<usize>) {
    let event_count = parts.len() * 2;
    let mut x_events = Vec::with_capacity(event_count);
    for (part_index, part) in parts.iter().enumerate() {
        if bounding_rect.intersects(&part.rect) {
            //if rect_contains_rect(&part.rect, bounding_rect) {
            //    cover_parts.push(part_index);
            //    continue;
            //}

            let px0 = part.rect.origin.x.max(bounding_rect.origin.x);
            let px1 = (part.rect.origin.x + part.rect.size.width).min(bounding_rect.origin.x + bounding_rect.size.width);
            debug_assert!(px0 != px1);

            x_events.push(Event::begin(px0, part_index));
            x_events.push(Event::end(px1, part_index));
        }
    }
    x_events.sort();

    let mut open_rects: Vec<OpenRect> = Vec::new();

    for xe in x_events {
        let rect = &parts[xe.key].rect;

        match xe.kind {
            EventKind::Begin => {
                let py0 = rect.origin.y.max(bounding_rect.origin.y);
                let py1 = (rect.origin.y + rect.size.height).min(bounding_rect.origin.y + bounding_rect.size.height);
                debug_assert!(py0 != py1);

                let mut y_events = Vec::new();
                y_events.push(Event::begin(py0, xe.key));
                y_events.push(Event::end(py1, xe.key));

                let mut new_open_rects = Vec::new();
                for open_rect in open_rects.drain(..) {
                    if open_rect.y0 < py1 && py0 < open_rect.y1 {
                        if xe.value.to_f32_px() > open_rect.x0 {
                            f(Rect::from_points(open_rect.x0,
                                                open_rect.y0,
                                                xe.value.to_f32_px(),
                                                open_rect.y1),
                              &open_rect.keys);
                        }
                        for key in open_rect.keys {
                            y_events.push(Event::begin(open_rect.y0, key));
                            y_events.push(Event::end(open_rect.y1, key));
                        }
                    } else {
                        new_open_rects.push(open_rect);
                    }
                }
                open_rects = new_open_rects;

                y_events.sort();
                let mut active_y = HashSet::new();
                let mut y0 = y_events[0].value;

                for i in 0..y_events.len() {
                    let ye = &y_events[i];

                    if i == y_events.len()-1 || (i > 0 && ye.value > y_events[i-1].value) {
                        if active_y.len() > 0 && ye.value > y0 {
                            open_rects.push(OpenRect::new(active_y.clone(), xe.value.to_f32_px(), y0.to_f32_px(), ye.value.to_f32_px()));
                        }
                        y0 = ye.value;
                    }

                    match ye.kind {
                        EventKind::Begin => {
                            active_y.insert(ye.key);
                        }
                        EventKind::End => {
                            active_y.remove(&ye.key);
                        }
                    }
                }
            }
            EventKind::End => {
                let mut new_open_rects = Vec::new();
                for mut rect in open_rects.drain(..) {
                    if rect.keys.contains(&xe.key) {
                        if xe.value.to_f32_px() > rect.x0 {
                            f(Rect::from_points(rect.x0,
                                                rect.y0,
                                                xe.value.to_f32_px(),
                                                rect.y1),
                              &rect.keys);
                        }
                        rect.keys.remove(&xe.key);
                        rect.x0 = xe.value.to_f32_px();
                        if rect.keys.len() > 0 {
                            new_open_rects.push(rect);
                        }
                    } else {
                        new_open_rects.push(rect);
                    }
                }
                open_rects = new_open_rects;
            }
        }
    }
}

const MAX_PRIMITIVES_PER_PASS: usize = 4;
const INVALID_PRIM_INDEX: u32 = 0xffffffff;
const INVALID_CLIP_INDEX: ClipIndex = ClipIndex(0xffffffff);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ClipIndex(u32);

// TODO (gw): Profile and create a smaller layout for simple passes if worthwhile...
#[derive(Debug)]
pub struct PackedDrawCommand {
    pub tile_p0: Point2D<f32>,
    pub tile_p1: Point2D<f32>,
    pub info: [u32; 4],
    pub prim_indices: [u32; MAX_PRIMITIVES_PER_PASS],
}

impl PackedDrawCommand {
    fn set_primitive(&mut self, cmd_index: usize, prim_index: usize) {
        self.prim_indices[cmd_index] = prim_index as u32;
    }

    fn new(tile_rect: &Rect<f32>, layer_index_in_ubo: u32) -> PackedDrawCommand {
        PackedDrawCommand {
            tile_p0: tile_rect.origin,
            tile_p1: tile_rect.bottom_right(),
            prim_indices: [INVALID_PRIM_INDEX; MAX_PRIMITIVES_PER_PASS],
            info: [layer_index_in_ubo, 0, 0, 0],
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Eq, PartialEq, Copy, Clone, Hash)]
pub enum PrimitiveShader {
    Error,
    Prim1,
    Prim2,
    Prim3,
}

#[derive(Debug, Eq, PartialEq, Hash)]
pub struct BatchKey {
    pub shader: PrimitiveShader,
    pub prim_ubo_index: usize,
}

impl BatchKey {
    fn new(shader: PrimitiveShader, prim_ubo_index: usize) -> BatchKey {
        BatchKey {
            shader: shader,
            prim_ubo_index: prim_ubo_index,
        }
    }
}

#[derive(Debug)]
pub struct Ubo<KEY: Eq + Hash, TYPE> {
    pub items: Vec<TYPE>,
    map: HashMap<KEY, usize, BuildHasherDefault<FnvHasher>>,
}

impl<KEY: Eq + Hash + Copy, TYPE: Clone> Ubo<KEY, TYPE> {
    fn new() -> Ubo<KEY, TYPE> {
        Ubo {
            items: Vec::new(),
            map: HashMap::with_hasher(Default::default()),
        }
    }

    fn maybe_insert_and_get_index(&mut self, key: KEY, data: &TYPE) -> u32 {
        let map = &mut self.map;
        let items = &mut self.items;

        *map.entry(key).or_insert_with(|| {
            let index = items.len();
            items.push(data.clone());
            index
        }) as u32
    }
}

pub struct Tile {
    layer_index: usize,
    node_index: usize,
    pub result: Option<CompiledTile>,
    screen_rect: TransformedRect,
    pipeline_id: PipelineId,
}

pub struct CompiledTile {
    pub parts: Vec<PrimitivePart>,
    pub batches: HashMap<BatchKey, Vec<PackedDrawCommand>>,
}

impl CompiledTile {
    fn new() -> CompiledTile {
        CompiledTile {
            parts: Vec::new(),
            batches: HashMap::new(),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PrimitivePartIndex(u32);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PrimitiveIndex(u32);

impl PrimitiveIndex {
    fn new(index: usize) -> PrimitiveIndex {
        PrimitiveIndex(index as u32)
    }
}

struct RectanglePrimitive {
    color: ColorF,
}

struct TextPrimitive {
    color: ColorF,
    font_key: FontKey,
    size: Au,
    blur_radius: Au,
    glyph_range: ItemRange,
    run_key: TextRunKey,
}

struct ImagePrimitive {
    image_key: ImageKey,
    image_rendering: ImageRendering,
}

struct GradientPrimitive {
    stops_range: ItemRange,
    dir: AxisDirection,
}

struct BorderPrimitive {
    tl_outer: Point2D<f32>,
    tl_inner: Point2D<f32>,
    tr_outer: Point2D<f32>,
    tr_inner: Point2D<f32>,
    bl_outer: Point2D<f32>,
    bl_inner: Point2D<f32>,
    br_outer: Point2D<f32>,
    br_inner: Point2D<f32>,
    left_width: f32,
    top_width: f32,
    right_width: f32,
    bottom_width: f32,
    radius: BorderRadius,
    left_color: ColorF,
    top_color: ColorF,
    right_color: ColorF,
    bottom_color: ColorF,
}

struct BoxShadowPrimitive {
    box_bounds: Rect<f32>,
    color: ColorF,
    blur_radius: f32,
    spread_radius: f32,
    border_radius: f32,
    clip_mode: BoxShadowClipMode,
}

enum PrimitiveDetails {
    Rectangle(RectanglePrimitive),
    Text(TextPrimitive),
    Image(ImagePrimitive),
    Border(BorderPrimitive),
    Gradient(GradientPrimitive),
    BoxShadow(BoxShadowPrimitive),
}

struct Primitive {
    rect: Rect<f32>,
    clip: Option<ClipIndex>,
    details: PrimitiveDetails,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct LayerTemplateIndex(u32);

#[derive(Clone, Debug)]
pub struct PackedLayer {
    transform: Matrix4D<f32>,
    inv_transform: Matrix4D<f32>,
    screen_vertices: [Point4D<f32>; 4],
    blend_info: [f32; 4],
}

#[derive(Debug)]
struct TransformedRect {
    local_rect: Rect<f32>,
    vertices: [Point4D<f32>; 4],
    screen_rect: Rect<f32>,
}

impl TransformedRect {
    fn new(rect: &Rect<f32>, transform: &Matrix4D<f32>) -> TransformedRect {
        let vertices = [
            transform.transform_point4d(&Point4D::new(rect.origin.x,
                                                      rect.origin.y,
                                                      0.0,
                                                      1.0)),
            transform.transform_point4d(&Point4D::new(rect.bottom_left().x,
                                                      rect.bottom_left().y,
                                                      0.0,
                                                      1.0)),
            transform.transform_point4d(&Point4D::new(rect.bottom_right().x,
                                                      rect.bottom_right().y,
                                                      0.0,
                                                      1.0)),
            transform.transform_point4d(&Point4D::new(rect.top_right().x,
                                                      rect.top_right().y,
                                                      0.0,
                                                      1.0)),
        ];

        let mut screen_min: Point2D<f32> = Point2D::new( 1000000.0,  1000000.0);
        let mut screen_max: Point2D<f32> = Point2D::new(-1000000.0, -1000000.0);

        for vertex in &vertices {
            let inv_w = 1.0 / vertex.w;
            let vx: f32 = vertex.x * inv_w;
            let vy: f32 = vertex.y * inv_w;
            screen_min.x = screen_min.x.min(vx);
            screen_min.y = screen_min.y.min(vy);
            screen_max.x = screen_max.x.max(vx);
            screen_max.y = screen_max.y.max(vy);
        }

        TransformedRect {
            local_rect: *rect,
            vertices: vertices,
            screen_rect: Rect::new(screen_min, Size2D::new(screen_max.x - screen_min.x, screen_max.y - screen_min.y)),
        }
    }
}

struct LayerTemplate {
    packed: PackedLayer,
    pipeline_id: PipelineId,
    primitives: Vec<Primitive>,
    quadtree: Quadtree<PrimitiveIndex>,
    device_pixel_ratio: f32,
    index_in_ubo: u32,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum CornerKind {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum RectangleKind {
    Solid,
    HorizontalGradient,
    VerticalGradient,
    //BorderCorner,
    BoxShadowEdge,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Opacity {
    Opaque,
    Translucent,
}

impl Opacity {
    fn from_bool(is_opaque: bool) -> Opacity {
        if is_opaque {
            Opacity::Opaque
        } else {
            Opacity::Translucent
        }
    }

    fn from_color(color: &ColorF) -> Opacity {
        if color.a == 1.0 {
            Opacity::Opaque
        } else {
            Opacity::Translucent
        }
    }

    fn from_colors(colors: &[&ColorF]) -> Opacity {
        for color in colors {
            if color.a < 1.0 {
                return Opacity::Translucent;
            }
        }

        Opacity::Opaque
    }

    fn from_color_and_texture(color: &ColorF, texture_is_opaque: bool) -> Opacity {
        if color.a == 1.0 && texture_is_opaque {
            Opacity::Opaque
        } else {
            Opacity::Translucent
        }
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum RotationKind {
    Angle0,
    Angle90,
    Angle180,
    Angle270,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum PrimitiveKind {
    Rectangle = 0,
    Image,
    Text,
    BorderCorner,
}

struct PrimitivePartList {
    //color_texture_id: TextureId,
    //mask_texture_id: TextureId,
    parts: Vec<PrimitivePart>,
}

impl PrimitivePartList {
    fn new() -> PrimitivePartList {
        PrimitivePartList {
            //color_texture_id: TextureId(0),
            //mask_texture_id: TextureId(0),
            parts: Vec::new(),
        }
    }

    #[inline]
    fn push_parts(&mut self,
                  parts: &[PrimitivePart],
                  clip: Option<&Clip>) {
        for part in parts {
            self.push_part(part, clip);
        }
    }

    #[inline]
    fn push_part(&mut self,
                 part: &PrimitivePart,
                 clip: Option<&Clip>) {
        match clip {
            Some(clip) => {
                self.parts.extend_from_slice(&part.clip(clip));
            }
            None => {
                self.parts.push(part.clone());
            }
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct ClipInfo {
    ref_point: Point2D<f32>,
    width: Point2D<f32>,
    outer_radius: Point2D<f32>,
    inner_radius: Point2D<f32>,
}

impl ClipInfo {
    fn new(ref_point: Point2D<f32>,
           width: Point2D<f32>,
           outer_radius: Point2D<f32>,
           inner_radius: Point2D<f32>) -> ClipInfo {
        ClipInfo {
            ref_point: ref_point,
            width: width,
            outer_radius: outer_radius,
            inner_radius: inner_radius,
        }
    }

    fn invalid() -> ClipInfo {
        ClipInfo {
            ref_point: Point2D::zero(),
            width: Point2D::zero(),
            outer_radius: Point2D::new(f32::MAX, f32::MAX),
            inner_radius: Point2D::zero(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct PrimitivePart {
    pub rect: Rect<f32>,
    pub st0: Point2D<f32>,
    pub st1: Point2D<f32>,
    pub color0: ColorF,
    pub color1: ColorF,
    pub kind: PrimitiveKind,
    pub rect_kind: RectangleKind,
    pub rotation: RotationKind,
    pub opacity: Opacity,
    pub clip_info: ClipInfo,
}

impl PrimitivePart {
    fn solid(origin: Point2D<f32>,
             size: Size2D<f32>,
             color: ColorF,
             clip_info: Option<ClipInfo>) -> PrimitivePart {
        let opacity = match clip_info {
            Some(..) => Opacity::Translucent,
            None => Opacity::from_color(&color),
        };

        PrimitivePart {
            rect: Rect::new(origin, size),
            st0: Point2D::zero(),
            st1: Point2D::zero(),
            color0: color,
            color1: color,
            kind: PrimitiveKind::Rectangle,
            rect_kind: RectangleKind::Solid,
            rotation: RotationKind::Angle0,
            opacity: opacity,
            clip_info: clip_info.unwrap_or(ClipInfo::invalid()),
        }
    }

    fn text(origin: Point2D<f32>,
            size: Size2D<f32>,
            st0: Point2D<f32>,
            st1: Point2D<f32>,
            color: ColorF) -> PrimitivePart {
        PrimitivePart {
            rect: Rect::new(origin, size),
            st0: st0,
            st1: st1,
            color0: color,
            color1: color,
            kind: PrimitiveKind::Text,
            rect_kind: RectangleKind::Solid,
            rotation: RotationKind::Angle0,
            opacity: Opacity::Translucent,
            clip_info: ClipInfo::invalid(),
        }
    }

    fn image(origin: Point2D<f32>,
             size: Size2D<f32>,
             st0: Point2D<f32>,
             st1: Point2D<f32>,
             opacity: Opacity) -> PrimitivePart {
        let color = ColorF::new(1.0, 1.0, 1.0, 1.0);
        PrimitivePart {
            rect: Rect::new(origin, size),
            st0: st0,
            st1: st1,
            color0: color,
            color1: color,
            kind: PrimitiveKind::Image,
            rect_kind: RectangleKind::Solid,
            rotation: RotationKind::Angle0,
            opacity: opacity,
            clip_info: ClipInfo::invalid(),
        }
    }

    fn gradient(origin: Point2D<f32>,
                size: Size2D<f32>,
                color0: ColorF,
                color1: ColorF,
                rect_kind: RectangleKind) -> PrimitivePart {
        PrimitivePart {
            rect: Rect::new(origin, size),
            st0: Point2D::zero(),
            st1: Point2D::zero(),
            color0: color0,
            color1: color1,
            kind: PrimitiveKind::Rectangle,
            rect_kind: rect_kind,
            rotation: RotationKind::Angle0,
            opacity: Opacity::from_colors(&[&color0, &color1]),
            clip_info: ClipInfo::invalid(),
        }
    }

    fn bc(origin: Point2D<f32>,
          size: Size2D<f32>,
          color0: ColorF,
          color1: ColorF,
          rotation: RotationKind,
          clip_info: Option<ClipInfo>) -> PrimitivePart {
        let opacity = match clip_info {
            Some(..) => Opacity::Translucent,
            None => Opacity::from_colors(&[&color0, &color1]),
        };

        PrimitivePart {
            rect: Rect::new(origin, size),
            st0: Point2D::zero(),
            st1: Point2D::zero(),
            color0: color0,
            color1: color1,
            kind: PrimitiveKind::BorderCorner,
            rect_kind: RectangleKind::Solid,
            rotation: rotation,
            opacity: opacity,
            clip_info: clip_info.unwrap_or(ClipInfo::invalid()),
        }
    }

/*
    fn box_shadow(rect: Rect<f32>,
                  color: ColorF,
                  rotation: RotationKind) -> PrimitivePart {
        PrimitivePart {
            rect: rect,
            st0: Point2D::zero(),
            st1: Point2D::zero(),
            color0: color,
            color1: color,
            kind: PrimitiveKind::Rectangle,
            rect_kind: RectangleKind::Solid,
            rotation: rotation,
            opacity: Opacity::from_color(&color),
            clip_index: INVALID_CLIP_INDEX,
            corner_kind: CornerKind::TopLeft,
            padding: [0, 0],
        }
    }
*/

    fn box_shadow_texture(rect: Rect<f32>,
                          color: ColorF,
                          st0: Point2D<f32>,
                          st1: Point2D<f32>,
                          texture_is_opaque: bool,
                          rotation: RotationKind,
                          rect_kind: RectangleKind) -> PrimitivePart {
        PrimitivePart {
            rect: rect,
            st0: st0,
            st1: st1,
            color0: color,
            color1: color,
            kind: PrimitiveKind::Image,
            rect_kind: rect_kind,
            rotation: rotation,
            opacity: Opacity::from_color_and_texture(&color, texture_is_opaque),
            clip_info: ClipInfo::invalid(),
        }
    }

    fn interp(&self, clipped_rect: &Rect<f32>) -> PrimitivePart {
        debug_assert!(clipped_rect.size.width > 0.0);
        debug_assert!(clipped_rect.size.height > 0.0);

        let f0 = Point2D::new((clipped_rect.origin.x - self.rect.origin.x) / self.rect.size.width,
                              (clipped_rect.origin.y - self.rect.origin.y) / self.rect.size.height);

        let f1 = Point2D::new((clipped_rect.origin.x + clipped_rect.size.width - self.rect.origin.x) / self.rect.size.width,
                              (clipped_rect.origin.y + clipped_rect.size.height - self.rect.origin.y) / self.rect.size.height);

        let st0 = Point2D::new(util::lerp(self.st0.x, self.st1.x, f0.x),
                               util::lerp(self.st0.y, self.st1.y, f0.y));

        let st1 = Point2D::new(util::lerp(self.st0.x, self.st1.x, f1.x),
                               util::lerp(self.st0.y, self.st1.y, f1.y));

        // TODO: Need to do color bilerp in some cases too!?
        PrimitivePart {
            rect: *clipped_rect,
            st0: st0,
            st1: st1,
            color0: self.color0,
            color1: self.color1,
            kind: self.kind,
            rect_kind: self.rect_kind,
            rotation: self.rotation,
            opacity: self.opacity,
            clip_info: ClipInfo::invalid(),
        }
    }

    fn clip(&self, clip: &Clip) -> Vec<PrimitivePart> {
        let mut parts = Vec::new();
        let clip_rect = Rect::new(clip.p0, Size2D::new(clip.p1.x - clip.p0.x, clip.p1.y - clip.p0.y));

        match clip.clip_kind {
            ClipKind::ClipIn => {
                // TODO(gw): Support corner radii.
                let rects = subtract_rect(&self.rect, &clip_rect);

                for rect in rects {
                    parts.push(self.interp(&rect));
                }
            }
            ClipKind::ClipOut => {
                // TODO(gw): Support irregular radii...
                let r = clip.top_left.outer_radius_x;

                if r == 0.0 {
                    if clip_rect.contains_rect(&self.rect) {
                        // TODO(gw): Optimize common path to not include a heap allocation...
                        parts.push(self.clone());
                    } else {
                        let clipped_rect = clip_rect.intersection(&self.rect);

                        if let Some(clipped_rect) = clipped_rect {
                            parts.push(self.interp(&clipped_rect));
                        }
                    }
                } else {
                    let p0 = self.rect.origin;
                    let p1 = self.rect.bottom_right();

                    let inner_top = Rect::from_points(p0.x + r,
                                                      p0.y,
                                                      p1.x - r,
                                                      p0.y + r);

                    let inner_bottom = Rect::from_points(p0.x + r,
                                                         p1.y - r,
                                                         p1.x - r,
                                                         p1.y);

                    let inner_left = Rect::from_points(p0.x,
                                                       p0.y + r,
                                                       p0.x + r,
                                                       p1.y - r);

                    let inner_right = Rect::from_points(p1.x - r,
                                                        p0.y + r,
                                                        p1.x,
                                                        p1.y - r);

                    let inner_mid = Rect::from_points(p0.x + r,
                                                      p0.y + r,
                                                      p1.x - r,
                                                      p1.y - r);

                    if let Some(rect) = self.rect.intersection(&inner_top) {
                        parts.push(PrimitivePart::solid(rect.origin,
                                                        rect.size,
                                                        self.color0,
                                                        None));
                    }

                    if let Some(rect) = self.rect.intersection(&inner_left) {
                        parts.push(PrimitivePart::solid(rect.origin,
                                                        rect.size,
                                                        self.color0,
                                                        None));
                    }

                    if let Some(rect) = self.rect.intersection(&inner_right) {
                        parts.push(PrimitivePart::solid(rect.origin,
                                                        rect.size,
                                                        self.color0,
                                                        None));
                    }

                    if let Some(rect) = self.rect.intersection(&inner_bottom) {
                        parts.push(PrimitivePart::solid(rect.origin,
                                                        rect.size,
                                                        self.color0,
                                                        None));
                    }

                    if let Some(rect) = self.rect.intersection(&inner_mid) {
                        parts.push(PrimitivePart::solid(rect.origin,
                                                        rect.size,
                                                        self.color0,
                                                        None));
                    }

                    if let Some(rect) = self.rect.intersection(&clip.top_left.rect) {
                        let clip_info = ClipInfo::new(clip.top_left.rect.bottom_right(),
                                                      Point2D::zero(),
                                                      Point2D::new(clip.top_left.outer_radius_x, clip.top_left.outer_radius_y),
                                                      Point2D::new(clip.top_left.inner_radius_x, clip.top_left.inner_radius_y));
                        parts.push(PrimitivePart::solid(rect.origin,
                                                        rect.size,
                                                        self.color0,
                                                        Some(clip_info)));
                    }

                    if let Some(rect) = self.rect.intersection(&clip.top_right.rect) {
                        let clip_info = ClipInfo::new(clip.top_right.rect.bottom_left(),
                                                      Point2D::zero(),
                                                      Point2D::new(clip.top_right.outer_radius_x, clip.top_right.outer_radius_y),
                                                      Point2D::new(clip.top_right.inner_radius_x, clip.top_right.inner_radius_y));
                        parts.push(PrimitivePart::solid(rect.origin,
                                                        rect.size,
                                                        self.color0,
                                                        Some(clip_info)));
                    }

                    if let Some(rect) = self.rect.intersection(&clip.bottom_left.rect) {
                        let clip_info = ClipInfo::new(clip.bottom_left.rect.top_right(),
                                                      Point2D::zero(),
                                                      Point2D::new(clip.bottom_left.outer_radius_x, clip.bottom_left.outer_radius_y),
                                                      Point2D::new(clip.bottom_left.inner_radius_x, clip.bottom_left.inner_radius_y));
                        parts.push(PrimitivePart::solid(rect.origin,
                                                        rect.size,
                                                        self.color0,
                                                        Some(clip_info)));
                    }

                    if let Some(rect) = self.rect.intersection(&clip.bottom_right.rect) {
                        let clip_info = ClipInfo::new(clip.bottom_right.rect.origin,
                                                      Point2D::zero(),
                                                      Point2D::new(clip.bottom_right.outer_radius_x, clip.bottom_right.outer_radius_y),
                                                      Point2D::new(clip.bottom_right.inner_radius_x, clip.bottom_right.inner_radius_y));
                        parts.push(PrimitivePart::solid(rect.origin,
                                                        rect.size,
                                                        self.color0,
                                                        Some(clip_info)));
                    }
                }
            }
        }

        parts
    }
}

impl LayerTemplate {
    fn build_resource_list(&self,
                           device_pixel_ratio: f32,
                           prim_index_buffer: &Vec<PrimitiveIndex>,
                           text_buffer: &mut TextBuffer,
                           auxiliary_lists: &AuxiliaryLists) -> ResourceList {
        let mut resource_list = ResourceList::new(device_pixel_ratio);

        for prim_index in prim_index_buffer {
            let PrimitiveIndex(pi) = *prim_index;
            let prim = &self.primitives[pi as usize];

            match prim.details {
                PrimitiveDetails::Rectangle(..) => {}
                PrimitiveDetails::Gradient(..) => {}
                PrimitiveDetails::Border(..) => {}
                PrimitiveDetails::BoxShadow(ref details) => {
                    if details.blur_radius != 0.0 ||
                       details.spread_radius != 0.0 ||
                       details.clip_mode != BoxShadowClipMode::None {
                        resource_list.add_box_shadow_corner(details.blur_radius,
                                                            details.border_radius,
                                                            &prim.rect,
                                                            false);
                        resource_list.add_box_shadow_edge(details.blur_radius,
                                                          details.border_radius,
                                                          &prim.rect,
                                                          false);
                        if details.clip_mode == BoxShadowClipMode::Inset {
                            resource_list.add_box_shadow_corner(details.blur_radius,
                                                                details.border_radius,
                                                                &prim.rect,
                                                                true);
                            resource_list.add_box_shadow_edge(details.blur_radius,
                                                              details.border_radius,
                                                              &prim.rect,
                                                              true);
                        }
                    }
                }
                PrimitiveDetails::Text(ref details) => {
                    let glyphs = auxiliary_lists.glyph_instances(&details.glyph_range);
                    for glyph in glyphs {
                        let glyph = Glyph::new(details.size, details.blur_radius, glyph.index);
                        resource_list.add_glyph(details.font_key, glyph);
                    }
                    text_buffer.push_text(details.run_key,
                                          details.glyph_range,
                                          details.font_key,
                                          details.size,
                                          details.blur_radius);
                }
                PrimitiveDetails::Image(ref details) => {
                    resource_list.add_image(details.image_key,
                                            details.image_rendering);
                }
            }
        }

        resource_list
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ClipKind {
    ClipOut,
    ClipIn,
}

#[derive(Clone, Debug)]
pub struct PackedGlyph {
    pub p0: Point2D<f32>,
    pub p1: Point2D<f32>,
    pub st0: Point2D<f32>,
    pub st1: Point2D<f32>,
}

#[derive(Debug, Clone)]
pub struct TextRun {
    glyphs: ItemRange,
    key: FontKey,
    size: Au,
    blur_radius: Au,

    texture_id: TextureId,
    pub st0: Point2D<f32>,
    pub st1: Point2D<f32>,
    pub rect: Rect<f32>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct TextRunKey(usize);

pub struct TextBuffer {
    pub texture_size: f32,
    pub page_allocator: TexturePage,
    pub glyphs: Vec<PackedGlyph>,
    runs: HashMap<TextRunKey, TextRun>,
}

impl TextBuffer {
    fn new(size: u32) -> TextBuffer {
        TextBuffer {
            texture_size: size as f32,
            page_allocator: TexturePage::new(TextureId(0), size),
            glyphs: Vec::new(),
            runs: HashMap::new(),
        }
    }

    fn push_text(&mut self,
                 key: TextRunKey,
                 glyphs: ItemRange,
                 font_key: FontKey,
                 size: Au,
                 blur_radius: Au) {
        self.runs.insert(key, TextRun {
            glyphs: glyphs,
            key: font_key,
            size: size,
            blur_radius: blur_radius,
            st0: Point2D::zero(),
            st1: Point2D::zero(),
            rect: Rect::new(Point2D::zero(), Size2D::zero()),
            texture_id: TextureId(0),
        });
    }

    fn build(&mut self,
             resource_cache: &ResourceCache,
             auxiliary_lists: &AuxiliaryLists,
             frame_id: FrameId,
             device_pixel_ratio: f32) {
        for (_, run) in &mut self.runs {
            // TODO(gw): This is a total hack to make it possible to call build() multiple times
            //           per frame - and still work with auxiliary_lists per pipeline. Clean this up!!
            if run.texture_id == TextureId(0) {
                let src_glyphs = auxiliary_lists.glyph_instances(&run.glyphs);
                let mut glyph_key = GlyphKey::new(run.key,
                                                  run.size,
                                                  run.blur_radius,
                                                  src_glyphs[0].index);
                let blur_offset = run.blur_radius.to_f32_px() * (BLUR_INFLATION_FACTOR as f32) / 2.0;
                let mut glyphs = Vec::new();

                for glyph in src_glyphs {
                    glyph_key.index = glyph.index;
                    let image_info = resource_cache.get_glyph(&glyph_key, frame_id);
                    if let Some(image_info) = image_info {
                        // TODO(gw): Need a general solution to handle multiple texture pages per tile in WR2!
                        assert!(run.texture_id == TextureId(0) ||
                                run.texture_id == image_info.texture_id);
                        run.texture_id = image_info.texture_id;

                        let x = glyph.x + image_info.user_data.x0 as f32 / device_pixel_ratio - blur_offset;
                        let y = glyph.y - image_info.user_data.y0 as f32 / device_pixel_ratio - blur_offset;

                        let width = image_info.requested_rect.size.width as f32 / device_pixel_ratio;
                        let height = image_info.requested_rect.size.height as f32 / device_pixel_ratio;

                        let uv_rect = image_info.uv_rect();

                        glyphs.push(PackedGlyph {
                            p0: Point2D::new(x, y),
                            p1: Point2D::new(x + width, y + height),
                            st0: uv_rect.top_left,
                            st1: uv_rect.bottom_right,
                        });
                    }
                }

                if !glyphs.is_empty() {
                    let mut rect = Rect::zero();
                    for glyph in &glyphs {
                        rect = rect.union(&Rect::new(glyph.p0, Size2D::new(glyph.p1.x - glyph.p0.x, glyph.p1.y - glyph.p0.y)));
                    }

                    let size = Size2D::new(rect.size.width.ceil() as u32, rect.size.height.ceil() as u32);

                    let origin = self.page_allocator
                                     .allocate(&size, TextureFilter::Linear)
                                     .expect("handle no texture space!");

                    run.st0 = Point2D::new(origin.x as f32 / self.texture_size,
                                           origin.y as f32 / self.texture_size);
                    run.st1 = Point2D::new((origin.x + size.width) as f32 / self.texture_size,
                                           (origin.y + size.height) as f32 / self.texture_size);
                    run.rect = rect;

                    let d = Point2D::new(origin.x as f32, origin.y as f32) - rect.origin;
                    for glyph in &glyphs {
                        self.glyphs.push(PackedGlyph {
                            st0: glyph.st0,
                            st1: glyph.st1,
                            p0: glyph.p0 + d,
                            p1: glyph.p1 + d,
                        });
                    }
                }
            }
        }
    }

    fn get_text(&self, key: TextRunKey) -> &TextRun {
        &self.runs[&key]
    }
}

#[derive(Debug, Clone)]
pub struct ClipCorner {
    rect: Rect<f32>,
    outer_radius_x: f32,
    outer_radius_y: f32,
    inner_radius_x: f32,
    inner_radius_y: f32,
}

impl ClipCorner {
    fn invalid() -> ClipCorner {
        ClipCorner {
            rect: Rect::new(Point2D::zero(), Size2D::zero()),
            outer_radius_x: 0.0,
            outer_radius_y: 0.0,
            inner_radius_x: 0.0,
            inner_radius_y: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Clip {
    p0: Point2D<f32>,
    p1: Point2D<f32>,
    clip_kind: ClipKind,
    padding: [u32; 3],
    top_left: ClipCorner,
    top_right: ClipCorner,
    bottom_left: ClipCorner,
    bottom_right: ClipCorner,
}

impl Clip {
    /*
    pub fn get_rect(&self) -> Rect<f32> {
        Rect::new(self.p0, Size2D::new(self.p1.x - self.p0.x,
                                       self.p1.y - self.p0.y))
    }*/

    pub fn from_clip_region(clip: &ComplexClipRegion, clip_kind: ClipKind) -> Clip {
        Clip {
            p0: clip.rect.origin,
            p1: clip.rect.bottom_right(),
            clip_kind: clip_kind,
            padding: [0, 0, 0],
            top_left: ClipCorner {
                rect: Rect::new(Point2D::new(clip.rect.origin.x, clip.rect.origin.y),
                                Size2D::new(clip.radii.top_left.width, clip.radii.top_left.height)),
                outer_radius_x: clip.radii.top_left.width,
                outer_radius_y: clip.radii.top_left.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            top_right: ClipCorner {
                rect: Rect::new(Point2D::new(clip.rect.origin.x + clip.rect.size.width - clip.radii.top_right.width,
                                             clip.rect.origin.y),
                                Size2D::new(clip.radii.top_right.width, clip.radii.top_right.height)),
                outer_radius_x: clip.radii.top_right.width,
                outer_radius_y: clip.radii.top_right.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            bottom_left: ClipCorner {
                rect: Rect::new(Point2D::new(clip.rect.origin.x,
                                             clip.rect.origin.y + clip.rect.size.height - clip.radii.bottom_left.height),
                                Size2D::new(clip.radii.bottom_left.width, clip.radii.bottom_left.height)),
                outer_radius_x: clip.radii.bottom_left.width,
                outer_radius_y: clip.radii.bottom_left.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            bottom_right: ClipCorner {
                rect: Rect::new(Point2D::new(clip.rect.origin.x + clip.rect.size.width - clip.radii.bottom_right.width,
                                             clip.rect.origin.y + clip.rect.size.height - clip.radii.bottom_right.height),
                                Size2D::new(clip.radii.bottom_right.width, clip.radii.bottom_right.height)),
                outer_radius_x: clip.radii.bottom_right.width,
                outer_radius_y: clip.radii.bottom_right.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
        }
    }

    pub fn from_rect(rect: &Rect<f32>, clip_kind: ClipKind) -> Clip {
        Clip {
            p0: rect.origin,
            p1: rect.bottom_right(),
            clip_kind: clip_kind,
            padding: [0, 0, 0],
            top_left: ClipCorner::invalid(),
            top_right: ClipCorner::invalid(),
            bottom_left: ClipCorner::invalid(),
            bottom_right: ClipCorner::invalid(),
        }
    }

    pub fn from_border_radius(rect: &Rect<f32>,
                              outer_radius: &BorderRadius,
                              inner_radius: &BorderRadius,
                              clip_kind: ClipKind) -> Clip {
        Clip {
            p0: rect.origin,
            p1: rect.bottom_right(),
            clip_kind: clip_kind,
            padding: [0, 0, 0],
            top_left: ClipCorner {
                rect: Rect::new(Point2D::new(rect.origin.x, rect.origin.y),
                                Size2D::new(outer_radius.top_left.width, outer_radius.top_left.height)),
                outer_radius_x: outer_radius.top_left.width,
                outer_radius_y: outer_radius.top_left.height,
                inner_radius_x: inner_radius.top_left.width,
                inner_radius_y: inner_radius.top_left.height,
            },
            top_right: ClipCorner {
                rect: Rect::new(Point2D::new(rect.origin.x + rect.size.width - outer_radius.top_right.width,
                                             rect.origin.y),
                                Size2D::new(outer_radius.top_right.width, outer_radius.top_right.height)),
                outer_radius_x: outer_radius.top_right.width,
                outer_radius_y: outer_radius.top_right.height,
                inner_radius_x: inner_radius.top_right.width,
                inner_radius_y: inner_radius.top_right.height,
            },
            bottom_left: ClipCorner {
                rect: Rect::new(Point2D::new(rect.origin.x,
                                             rect.origin.y + rect.size.height - outer_radius.bottom_left.height),
                                Size2D::new(outer_radius.bottom_left.width, outer_radius.bottom_left.height)),
                outer_radius_x: outer_radius.bottom_left.width,
                outer_radius_y: outer_radius.bottom_left.height,
                inner_radius_x: inner_radius.bottom_left.width,
                inner_radius_y: inner_radius.bottom_left.height,
            },
            bottom_right: ClipCorner {
                rect: Rect::new(Point2D::new(rect.origin.x + rect.size.width - outer_radius.bottom_right.width,
                                             rect.origin.y + rect.size.height - outer_radius.bottom_right.height),
                                Size2D::new(outer_radius.bottom_right.width, outer_radius.bottom_right.height)),
                outer_radius_x: outer_radius.bottom_right.width,
                outer_radius_y: outer_radius.bottom_right.height,
                inner_radius_x: inner_radius.bottom_right.width,
                inner_radius_y: inner_radius.bottom_right.height,
            },
        }
    }

/*
    pub fn affects(&self, rect: &Rect<f32>) -> bool {
        let clip_rect = self.get_rect();

        // If rect isn't intersecting the clip rect, then it will have no effect.
        if clip_rect.intersects(rect) {
            if rect_contains_rect(&clip_rect, rect) {
                // If rect intersects with any of the border corner rects, then it will
                // have an effect (this is a conservative test, but should catch most cases).
                if self.top_left.rect.intersects(rect) {
                    return true;
                }

                if self.top_right.rect.intersects(rect) {
                    return true;
                }

                if self.bottom_left.rect.intersects(rect) {
                    return true;
                }

                if self.bottom_right.rect.intersects(rect) {
                    return true;
                }

                return false;
            } else {
                return true;
            }
        } else {
            return false
        }
    }*/
}

pub struct FrameBuilderConfig {

}

impl FrameBuilderConfig {
    pub fn new() -> FrameBuilderConfig {
        FrameBuilderConfig {

        }
    }
}

pub struct Frame {
    pub viewport_size: Size2D<u32>,
    pub layer_ubo: Ubo<LayerTemplateIndex, PackedLayer>, // TODO(gw): Handle batching this, in crazy case where layer count > ubo size!
    //pub clips: Vec<Clip>, // TODO(gw): Handle batching this, in crazy case where layer count > ubo size!
    pub tiles: Vec<Tile>,
    pub text_buffer: TextBuffer,
    pub color_texture_id: TextureId,
    pub mask_texture_id: TextureId,
    pub debug_rects: Vec<(usize, Rect<f32>)>,
}

pub struct FrameBuilder {
    screen_rect: Rect<i32>,
    layers: Vec<LayerTemplate>,
    layer_stack: Vec<LayerTemplateIndex>,
    color_texture_id: TextureId,
    mask_texture_id: TextureId,
    scroll_offset: Point2D<f32>,
    device_pixel_ratio: f32,
    clips: Vec<Clip>,
    clip_index: Option<ClipIndex>,
    next_text_run_key: usize,
}

impl FrameBuilder {
    pub fn new(viewport_size: Size2D<f32>,
               scroll_offset: Point2D<f32>,
               device_pixel_ratio: f32) -> FrameBuilder {
        let viewport_size = Size2D::new(viewport_size.width as i32, viewport_size.height as i32);
        FrameBuilder {
            screen_rect: Rect::new(Point2D::zero(), viewport_size),
            layers: Vec::new(),
            layer_stack: Vec::new(),
            color_texture_id: TextureId(0),
            mask_texture_id: TextureId(0),
            scroll_offset: scroll_offset,
            device_pixel_ratio: device_pixel_ratio,
            clips: Vec::new(),
            clip_index: None,
            next_text_run_key: 0,
        }
    }

    pub fn set_clip(&mut self, clip: Clip) {
        let clip_index = ClipIndex(self.clips.len() as u32);
        self.clip_index = Some(clip_index);
        self.clips.push(clip);
    }

    pub fn clear_clip(&mut self) {
        self.clip_index = None;
    }

    pub fn push_layer(&mut self,
                      rect: Rect<f32>,
                      transform: Matrix4D<f32>,
                      opacity: f32,
                      pipeline_id: PipelineId) {
        // TODO(gw): Not 3d transform correct!
        let scroll_transform = transform.translate(self.scroll_offset.x,
                                                   self.scroll_offset.y,
                                                   0.0);

        let layer_rect = TransformedRect::new(&rect, &transform);

        let template = LayerTemplate {
            packed: PackedLayer {
                inv_transform: scroll_transform.invert(),
                transform: scroll_transform,
                screen_vertices: layer_rect.vertices,
                blend_info: [opacity, 0.0, 0.0, 0.0],
            },
            primitives: Vec::new(),
            quadtree: Quadtree::new(rect, 5, 32),
            pipeline_id: pipeline_id,
            device_pixel_ratio: self.device_pixel_ratio,
            index_in_ubo: 0,
        };

        self.layer_stack.push(LayerTemplateIndex(self.layers.len() as u32));
        self.layers.push(template);
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
    }

    pub fn add_solid_rectangle(&mut self,
                               rect: &Rect<f32>,
                               color: &ColorF) {
        if color.a == 0.0 {
            return;
        }

        let prim = RectanglePrimitive {
            color: *color,
        };

        self.add_primitive(rect, PrimitiveDetails::Rectangle(prim));
    }

    pub fn add_border(&mut self,
                      rect: Rect<f32>,
                      border: &BorderDisplayItem) {
        let radius = &border.radius;
        let left = &border.left;
        let right = &border.right;
        let top = &border.top;
        let bottom = &border.bottom;

        if (left.style != BorderStyle::Solid && left.style != BorderStyle::None) ||
           (top.style != BorderStyle::Solid && top.style != BorderStyle::None) ||
           (bottom.style != BorderStyle::Solid && bottom.style != BorderStyle::None) ||
           (right.style != BorderStyle::Solid && right.style != BorderStyle::None) {
            println!("TODO: Other border styles {:?} {:?} {:?} {:?}", left.style, top.style, bottom.style, right.style);
            return;
        }

        let tl_outer = Point2D::new(rect.origin.x, rect.origin.y);
        let tl_inner = tl_outer + Point2D::new(radius.top_left.width.max(left.width),
                                               radius.top_left.height.max(top.width));

        let tr_outer = Point2D::new(rect.origin.x + rect.size.width, rect.origin.y);
        let tr_inner = tr_outer + Point2D::new(-radius.top_right.width.max(right.width),
                                               radius.top_right.height.max(top.width));

        let bl_outer = Point2D::new(rect.origin.x, rect.origin.y + rect.size.height);
        let bl_inner = bl_outer + Point2D::new(radius.bottom_left.width.max(left.width),
                                               -radius.bottom_left.height.max(bottom.width));

        let br_outer = Point2D::new(rect.origin.x + rect.size.width,
                                    rect.origin.y + rect.size.height);
        let br_inner = br_outer - Point2D::new(radius.bottom_right.width.max(right.width),
                                               radius.bottom_right.height.max(bottom.width));

        let left_color = left.border_color(1.0, 2.0/3.0, 0.3, 0.7);
        let top_color = top.border_color(1.0, 2.0/3.0, 0.3, 0.7);
        let right_color = right.border_color(2.0/3.0, 1.0, 0.7, 0.3);
        let bottom_color = bottom.border_color(2.0/3.0, 1.0, 0.7, 0.3);

        let prim = BorderPrimitive {
            tl_outer: tl_outer,
            tl_inner: tl_inner,
            tr_outer: tr_outer,
            tr_inner: tr_inner,
            bl_outer: bl_outer,
            bl_inner: bl_inner,
            br_outer: br_outer,
            br_inner: br_inner,
            radius: radius.clone(),
            left_width: left.width,
            top_width: top.width,
            bottom_width: bottom.width,
            right_width: right.width,
            left_color: left_color,
            top_color: top_color,
            bottom_color: bottom_color,
            right_color: right_color,
        };

        self.add_primitive(&rect, PrimitiveDetails::Border(prim));
    }

    fn add_primitive(&mut self,
                     rect: &Rect<f32>,
                     details: PrimitiveDetails) {
//        let _pf = hprof::enter("add_primitive");
        let current_layer = *self.layer_stack.last().unwrap();
        let LayerTemplateIndex(layer_index) = current_layer;
        let layer = &mut self.layers[layer_index as usize];
        let prim = Primitive {
            rect: *rect,
            clip: self.clip_index,
            details: details,
        };
        let prim_index = layer.primitives.len();
        layer.primitives.push(prim);

        let prims = &layer.primitives;
        let quadtree = &mut layer.quadtree;
        quadtree.insert(PrimitiveIndex::new(prim_index), &|pi| {
            let PrimitiveIndex(pi) = pi;
            prims[pi as usize].rect
        });
    }

    pub fn add_gradient(&mut self,
                        rect: Rect<f32>,
                        start_point: Point2D<f32>,
                        end_point: Point2D<f32>,
                        stops: ItemRange) {
        // Fast paths for axis-aligned gradients:
        if start_point.x == end_point.x {
            let rect = Rect::new(Point2D::new(rect.origin.x, start_point.y),
                                 Size2D::new(rect.size.width, end_point.y - start_point.y));
            let prim = GradientPrimitive {
                stops_range: stops,
                dir: AxisDirection::Vertical,
            };
            self.add_primitive(&rect, PrimitiveDetails::Gradient(prim));
        } else if start_point.y == end_point.y {
            let rect = Rect::new(Point2D::new(start_point.x, rect.origin.y),
                                 Size2D::new(end_point.x - start_point.x, rect.size.height));
            let prim = GradientPrimitive {
                stops_range: stops,
                dir: AxisDirection::Horizontal,
            };
            self.add_primitive(&rect, PrimitiveDetails::Gradient(prim));
        } else {
            //println!("TODO: Angle gradients! {:?} {:?} {:?}", start_point, end_point, stops);
        }
    }

    pub fn add_text(&mut self,
                    rect: Rect<f32>,
                    font_key: FontKey,
                    size: Au,
                    blur_radius: Au,
                    color: &ColorF,
                    glyph_range: ItemRange) {
        if color.a == 0.0 {
            return
        }

        let run_key = TextRunKey(self.next_text_run_key);
        self.next_text_run_key += 1;

        let prim = TextPrimitive {
            color: *color,
            font_key: font_key,
            size: size,
            blur_radius: blur_radius,
            glyph_range: glyph_range,
            run_key: run_key,
        };
        self.add_primitive(&rect, PrimitiveDetails::Text(prim));
    }

    pub fn add_box_shadow(&mut self,
                          box_bounds: &Rect<f32>,
                          box_offset: &Point2D<f32>,
                          color: &ColorF,
                          blur_radius: f32,
                          spread_radius: f32,
                          border_radius: f32,
                          clip_mode: BoxShadowClipMode) {
        if color.a == 0.0 {
            return
        }

        let rect = compute_box_shadow_rect(box_bounds, box_offset, spread_radius);

        let prim = BoxShadowPrimitive {
            box_bounds: *box_bounds,
            color: *color,
            blur_radius: blur_radius,
            spread_radius: spread_radius,
            border_radius: border_radius,
            clip_mode: clip_mode,
        };

        self.add_primitive(&rect, PrimitiveDetails::BoxShadow(prim));
    }

    pub fn add_image(&mut self,
                     rect: Rect<f32>,
                     _stretch_size: &Size2D<f32>,
                     image_key: ImageKey,
                     image_rendering: ImageRendering) {
        let prim = ImagePrimitive {
            image_key: image_key,
            image_rendering: image_rendering,
        };

        self.add_primitive(&rect, PrimitiveDetails::Image(prim));
    }

    fn build_prim_parts(&mut self,
                        layer_index: usize,
                        node_index: usize,
                        text_buffer: &TextBuffer,
                        auxiliary_lists: &AuxiliaryLists,
                        resource_cache: &ResourceCache,
                        frame_id: FrameId) -> PrimitivePartList {
        let layer = &self.layers[layer_index];
        let node = &layer.quadtree.nodes[node_index];
        let primitives = &layer.primitives;
        let prim_index_buffer = &node.items;

        let mut color_texture_id = TextureId(0);
        let mut prim_part_list = PrimitivePartList::new();

        for prim_index in prim_index_buffer {
            let PrimitiveIndex(pi) = *prim_index;
            let prim = &primitives[pi as usize];

            let clip = prim.clip.map(|ci| {
                let ClipIndex(ci) = ci;
                &self.clips[ci as usize]
            });

            match prim.details {
                PrimitiveDetails::Rectangle(ref details) => {
                    let part = PrimitivePart::solid(prim.rect.origin,
                                                    prim.rect.size,
                                                    details.color,
                                                    None);

                    prim_part_list.push_part(&part, clip);
                }
                PrimitiveDetails::Image(ref details) => {
                    let image_info = resource_cache.get_image(details.image_key,
                                                              details.image_rendering,
                                                              frame_id);
                    let uv_rect = image_info.uv_rect();

                    assert!(color_texture_id == TextureId(0) ||
                            color_texture_id == image_info.texture_id);
                    color_texture_id = image_info.texture_id;

                    let part = PrimitivePart::image(prim.rect.origin,
                                                    prim.rect.size,
                                                    uv_rect.top_left,
                                                    uv_rect.bottom_right,
                                                    Opacity::from_bool(image_info.is_opaque));
                    prim_part_list.push_part(&part, clip);
                }
                PrimitiveDetails::Border(ref d) => {
                    let inner_radius = BorderRadius {
                        top_left: Size2D::new(d.radius.top_left.width - d.left_width,
                                              d.radius.top_left.width - d.left_width),
                        top_right: Size2D::new(d.radius.top_right.width - d.right_width,
                                               d.radius.top_right.width - d.right_width),
                        bottom_left: Size2D::new(d.radius.bottom_left.width - d.left_width,
                                                 d.radius.bottom_left.width - d.left_width),
                        bottom_right: Size2D::new(d.radius.bottom_right.width - d.right_width,
                                                  d.radius.bottom_right.width - d.right_width),
                    };

                    let clip = Clip::from_border_radius(&prim.rect,
                                                        &d.radius,
                                                        &inner_radius,
                                                        ClipKind::ClipOut);

                    let ci_tl = if d.radius.top_left != Size2D::zero() {
                        Some(ClipInfo::new(clip.top_left.rect.bottom_right(),
                                           Point2D::zero(),
                                           Point2D::new(clip.top_left.outer_radius_x, clip.top_left.outer_radius_y),
                                           Point2D::new(clip.top_left.inner_radius_x, clip.top_left.inner_radius_y)))
                    } else {
                        None
                    };


                    let ci_tr = if d.radius.top_right != Size2D::zero() {
                        Some(ClipInfo::new(clip.top_right.rect.bottom_left(),
                                           Point2D::zero(),
                                           Point2D::new(clip.top_right.outer_radius_x, clip.top_right.outer_radius_y),
                                           Point2D::new(clip.top_right.inner_radius_x, clip.top_right.inner_radius_y)))
                    } else {
                        None
                    };

                    let ci_bl = if d.radius.bottom_left != Size2D::zero() {
                        Some(ClipInfo::new(clip.bottom_left.rect.top_right(),
                                           Point2D::zero(),
                                           Point2D::new(clip.bottom_left.outer_radius_x, clip.bottom_left.outer_radius_y),
                                           Point2D::new(clip.bottom_left.inner_radius_x, clip.bottom_left.inner_radius_y)))
                    } else {
                        None
                    };

                    let ci_br = if d.radius.bottom_right != Size2D::zero() {
                        Some(ClipInfo::new(clip.bottom_right.rect.origin,
                                           Point2D::zero(),
                                           Point2D::new(clip.bottom_right.outer_radius_x, clip.bottom_right.outer_radius_y),
                                           Point2D::new(clip.bottom_right.inner_radius_x, clip.bottom_right.inner_radius_y)))
                    } else {
                        None
                    };

                    let parts = [
                        PrimitivePart::solid(Point2D::new(d.tl_outer.x, d.tl_inner.y),
                                             Size2D::new(d.left_width, d.bl_inner.y - d.tl_inner.y),
                                             d.left_color,
                                             None),
                        PrimitivePart::solid(Point2D::new(d.tl_inner.x, d.tl_outer.y),
                                             Size2D::new(d.tr_inner.x - d.tl_inner.x, d.tr_outer.y + d.top_width - d.tl_outer.y),
                                             d.top_color,
                                             None),
                        PrimitivePart::solid(Point2D::new(d.br_outer.x - d.right_width, d.tr_inner.y),
                                             Size2D::new(d.right_width, d.br_inner.y - d.tr_inner.y),
                                             d.right_color,
                                             None),
                        PrimitivePart::solid(Point2D::new(d.bl_inner.x, d.bl_outer.y - d.bottom_width),
                                             Size2D::new(d.br_inner.x - d.bl_inner.x, d.br_outer.y - d.bl_outer.y + d.bottom_width),
                                             d.bottom_color,
                                             None),
                        PrimitivePart::bc(d.tl_outer,
                                          Size2D::new(d.tl_inner.x - d.tl_outer.x, d.tl_inner.y - d.tl_outer.y),
                                          d.left_color,
                                          d.top_color,
                                          RotationKind::Angle0,
                                          ci_tl),
                        PrimitivePart::bc(Point2D::new(d.tr_inner.x, d.tr_outer.y),
                                          Size2D::new(d.tr_outer.x - d.tr_inner.x, d.tr_inner.y - d.tr_outer.y),
                                          d.top_color,
                                          d.right_color,
                                          RotationKind::Angle90,
                                          ci_tr),
                        PrimitivePart::bc(d.br_inner,
                                          Size2D::new(d.br_outer.x - d.br_inner.x, d.br_outer.y - d.br_inner.y),
                                          d.right_color,
                                          d.bottom_color,
                                          RotationKind::Angle180,
                                          ci_br),
                        PrimitivePart::bc(Point2D::new(d.bl_outer.x, d.bl_inner.y),
                                          Size2D::new(d.bl_inner.x - d.bl_outer.x, d.bl_outer.y - d.bl_inner.y),
                                          d.bottom_color,
                                          d.left_color,
                                          RotationKind::Angle270,
                                          ci_bl),
                    ];

                    prim_part_list.push_parts(&parts, None);
                }
                PrimitiveDetails::BoxShadow(ref details) => {
                    let mut parts = Vec::new();

                    // Fast path.
                    if details.blur_radius == 0.0 &&
                       details.spread_radius == 0.0 &&
                       details.clip_mode == BoxShadowClipMode::None {
                        parts.push(PrimitivePart::solid(prim.rect.origin,
                                                        prim.rect.size,
                                                        details.color,
                                                        None));
                    } else {
                        let inverted = match details.clip_mode {
                            BoxShadowClipMode::Outset | BoxShadowClipMode::None => false,
                            BoxShadowClipMode::Inset => true,
                        };

                        let edge_image = match BoxShadowRasterOp::create_edge(details.blur_radius,
                                                                              details.border_radius,
                                                                              &prim.rect,
                                                                              inverted,
                                                                              self.device_pixel_ratio) {
                            Some(raster_item) => {
                                let raster_item = RasterItem::BoxShadow(raster_item);
                                resource_cache.get_raster(&raster_item, frame_id)
                            }
                            None => resource_cache.get_dummy_color_image(),
                        };

                        // TODO(gw): A hack for texture ids here - need a better solution!
                        assert!(color_texture_id == TextureId(0) ||
                                color_texture_id == edge_image.texture_id);
                        color_texture_id = edge_image.texture_id;

                        let corner_image = match BoxShadowRasterOp::create_corner(details.blur_radius,
                                                                                  details.border_radius,
                                                                                  &prim.rect,
                                                                                  inverted,
                                                                                  self.device_pixel_ratio) {
                            Some(raster_item) => {
                                let raster_item = RasterItem::BoxShadow(raster_item);
                                resource_cache.get_raster(&raster_item, frame_id)
                            }
                            None => resource_cache.get_dummy_color_image(),
                        };

                        // TODO(gw): A hack for texture ids here - need a better solution!
                        assert!(color_texture_id == TextureId(0) ||
                                color_texture_id == corner_image.texture_id);
                        color_texture_id = corner_image.texture_id;

                        let metrics = BoxShadowMetrics::new(&prim.rect,
                                                            details.border_radius,
                                                            details.blur_radius);

                        // Draw the corners.
                        add_box_shadow_corners(&metrics,
                                               &details.box_bounds,
                                               &details.color,
                                               &corner_image.uv_rect(),
                                               corner_image.is_opaque,
                                               &mut parts);

                        // Draw the sides.
                        add_box_shadow_sides(&metrics,
                                             &details.color,
                                             &edge_image.uv_rect(),
                                             edge_image.is_opaque,
                                             &mut parts);

                        match details.clip_mode {
                            BoxShadowClipMode::None => {
                                // Fill the center area.
                                //self.add_color_rectangle(box_bounds, color, resource_cache, frame_id);
                                panic!("todo");
                            }
                            BoxShadowClipMode::Outset => {
                                // Fill the center area.
                                if metrics.br_inner.x > metrics.tl_inner.x &&
                                        metrics.br_inner.y > metrics.tl_inner.y {
                                    let center_rect =
                                        Rect::new(metrics.tl_inner,
                                                  Size2D::new(metrics.br_inner.x - metrics.tl_inner.x,
                                                              metrics.br_inner.y - metrics.tl_inner.y));

                                    // FIXME(pcwalton): This assumes the border radius is zero. That is not always
                                    // the case!
                                    let clip_in = Clip::from_rect(&details.box_bounds, ClipKind::ClipIn);

                                    let part = PrimitivePart::solid(metrics.tl_inner,
                                                                    Size2D::new(metrics.br_inner.x - metrics.tl_inner.x,
                                                                                metrics.br_inner.y - metrics.tl_inner.y),
                                                                    details.color,
                                                                    None);

                                    prim_part_list.push_part(&part, Some(&clip_in));
                                }
                            }
                            BoxShadowClipMode::Inset => {
                                // Fill in the outsides.
                                panic!("todo");
/*                                self.fill_outside_area_of_inset_box_shadow(box_bounds,
                                                                           box_offset,
                                                                           color,
                                                                           blur_radius,
                                                                           spread_radius,
                                                                           border_radius,
                                                                           resource_cache,
                                                                           frame_id);*/
                            }
                        }
                    }

                    match details.clip_mode {
                        BoxShadowClipMode::None => {
                            prim_part_list.push_parts(&parts, None);
                        }
                        BoxShadowClipMode::Inset => {
                            let clip = Clip::from_rect(&details.box_bounds, ClipKind::ClipOut);
                            prim_part_list.push_parts(&parts, Some(&clip));
                        }
                        BoxShadowClipMode::Outset => {
                            let clip = Clip::from_rect(&details.box_bounds, ClipKind::ClipIn);
                            prim_part_list.push_parts(&parts, Some(&clip));
                        }
                    };

                }
                PrimitiveDetails::Gradient(ref details) => {
                    let mut parts = Vec::new();

                    let stops = auxiliary_lists.gradient_stops(&details.stops_range);
                    for i in 0..(stops.len() - 1) {
                        let (prev_stop, next_stop) = (&stops[i], &stops[i + 1]);
                        let piece_origin;
                        let piece_size;
                        let rect_kind;
                        match details.dir {
                            AxisDirection::Horizontal => {
                                let prev_x = util::lerp(prim.rect.origin.x, prim.rect.max_x(), prev_stop.offset);
                                let next_x = util::lerp(prim.rect.origin.x, prim.rect.max_x(), next_stop.offset);
                                piece_origin = Point2D::new(prev_x, prim.rect.origin.y);
                                piece_size = Size2D::new(next_x - prev_x, prim.rect.size.height);
                                rect_kind = RectangleKind::HorizontalGradient;
                            }
                            AxisDirection::Vertical => {
                                let prev_y = util::lerp(prim.rect.origin.y, prim.rect.max_y(), prev_stop.offset);
                                let next_y = util::lerp(prim.rect.origin.y, prim.rect.max_y(), next_stop.offset);
                                piece_origin = Point2D::new(prim.rect.origin.x, prev_y);
                                piece_size = Size2D::new(prim.rect.size.width, next_y - prev_y);
                                rect_kind = RectangleKind::VerticalGradient;
                            }
                        }

                        parts.push(PrimitivePart::gradient(piece_origin,
                                                           piece_size,
                                                           prev_stop.color,
                                                           next_stop.color,
                                                           rect_kind));
                    }

                    prim_part_list.push_parts(&parts, clip);
                }
                PrimitiveDetails::Text(ref details) => {
                    let run = text_buffer.get_text(details.run_key);

                    // TODO(gw): Need a general solution to handle multiple texture pages per tile in WR2!
                    assert!(color_texture_id == TextureId(0) ||
                            color_texture_id == run.texture_id);
                    color_texture_id = run.texture_id;

                    let part = PrimitivePart::text(run.rect.origin,
                                                   run.rect.size,
                                                   run.st0,
                                                   run.st1,
                                                   details.color);
                    prim_part_list.push_part(&part, clip);
                }
            }
        }

        if color_texture_id != TextureId(0) {
            assert!(self.color_texture_id == TextureId(0) ||
                    self.color_texture_id == color_texture_id);
            self.color_texture_id = color_texture_id;
        }

        prim_part_list
    }

    pub fn build(&mut self,
                 resource_cache: &mut ResourceCache,
                 frame_id: FrameId,
                 pipeline_auxiliary_lists: &HashMap<PipelineId, AuxiliaryLists, BuildHasherDefault<FnvHasher>>) -> Frame {
        let mut layer_ubo = Ubo::new();
        let mut text_buffer = TextBuffer::new(TEXT_TARGET_SIZE);
        let mut debug_rects = Vec::new();

        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            let layer_index = LayerTemplateIndex(layer_index as u32);
            layer.index_in_ubo = layer_ubo.maybe_insert_and_get_index(layer_index, &layer.packed);
        }

        let screen_rect = Rect::new(Point2D::zero(),
                                    Size2D::new(self.screen_rect.size.width as f32,
                                                self.screen_rect.size.height as f32));

        let mut tiles = Vec::new();
        for (layer_index, layer) in self.layers.iter().enumerate() {
            for (node_index, node) in layer.quadtree.nodes.iter().enumerate() {
                if node.is_leaf() {
                    let node_screen_rect = TransformedRect::new(&node.rect,
                                                                &layer.packed.transform);

                    if node_screen_rect.screen_rect.intersects(&screen_rect) {
                        tiles.push(Tile {
                            layer_index: layer_index,
                            node_index: node_index,
                            result: None,
                            screen_rect: node_screen_rect,
                            pipeline_id: layer.pipeline_id,
                        })
                    }
                }
            }
        }

        let mut tile_tree = Quadtree::new(screen_rect, 5, 8);

        for (tile_index, tile) in tiles.iter().enumerate() {
            tile_tree.insert(tile_index, &|key| {
                tiles[key].screen_rect.screen_rect
            });
        }

        // Pre-pass - set up resource lists and text buffer...
        for tile in &tiles {
            let auxiliary_lists = pipeline_auxiliary_lists.get(&tile.pipeline_id)
                                                          .expect("No auxiliary lists?!");

            // TODO(gw): To multi-thread resource list building - remove text buffer
            //           from here, and iterate the created resource lists sequentially as a post step...
            let layer = &self.layers[tile.layer_index];
            let node = &layer.quadtree.nodes[tile.node_index];
            let resource_list = layer.build_resource_list(self.device_pixel_ratio,
                                                          &node.items,
                                                          &mut text_buffer,
                                                          auxiliary_lists);

            // TODO(gw): Split up this loop into multiple passes as per WR1 and send to worker threads!
            resource_cache.add_resource_list(&resource_list, frame_id);

            resource_cache.raster_pending_glyphs(frame_id);
            text_buffer.build(resource_cache,
                              auxiliary_lists,
                              frame_id,
                              self.device_pixel_ratio);    // Relies on glyphs being rasterized for UV layout ...
        }

        for tile in &mut tiles {
            let mut compiled_tile = CompiledTile::new();

            let auxiliary_lists = pipeline_auxiliary_lists.get(&tile.pipeline_id)
                                                          .expect("No auxiliary lists?!");

            // TODO(gw): To multi-thread resource list building - remove text buffer
            //           from here, and iterate the created resource lists sequentially as a post step...

            // Now that glyphs are rasterized, it's safe to build the text runs.

            // Convert the primitives in this node into a list of packed prim parts.
            // TODO(gw): This can be trivially run on worker threads!
            let part_list = self.build_prim_parts(tile.layer_index,
                                                  tile.node_index,
                                                  &text_buffer,
                                                  auxiliary_lists,
                                                  resource_cache,
                                                  frame_id);

            let layer = &self.layers[tile.layer_index];
            let node = &layer.quadtree.nodes[tile.node_index];

            debug_rects.push((node.items.len(), layer.packed.transform.transform_rect(&node.rect)));

            let part_offset = compiled_tile.parts.len();

            let mut sap = Sap::new();
            sap.doit(&node.rect, &part_list.parts, |rect, mut part_indices| {
                part_indices.sort_by(|a, b| {
                    b.cmp(&a)
                });

                let opaque_index = part_indices.iter().position(|pi| {
                    let part = &part_list.parts[*pi];
                    part.opacity == Opacity::Opaque
                });

                if let Some(opaque_index) = opaque_index {
                    part_indices.truncate(opaque_index+1);
                }

                //debug_rects.push((indices.len(), layer.packed.transform.transform_rect(&rect)));

                let mut draw_cmd = PackedDrawCommand::new(rect, layer.index_in_ubo);

                let shader = if part_indices.len() <= MAX_PRIMITIVES_PER_PASS {
                    for (cmd_index, part_index) in part_indices.iter().enumerate() {
                        draw_cmd.set_primitive(cmd_index, part_offset + *part_index);
                    }

                    match part_indices.len() {
                        0 => panic!("wtf"),
                        1 => PrimitiveShader::Prim1,
                        2 => PrimitiveShader::Prim2,
                        3 => PrimitiveShader::Prim3,
                        _ => {
                            //println!("found {} indices :(", c);
                            PrimitiveShader::Error
                        }
                    }
                } else {
                    //println!("found {} indices2 :(", indices.len());
                    PrimitiveShader::Error
                };

                // TODO(gw): All kinds of broken here when >1 prim ubo is needed...!
                let key = BatchKey::new(shader, 0);
                let batch = compiled_tile.batches.entry(key).or_insert_with(|| {
                    Vec::new()
                });
                batch.push(draw_cmd);
            });

            compiled_tile.parts.extend_from_slice(&part_list.parts);

            tile.result = Some(compiled_tile);
        }

        Frame {
            viewport_size: Size2D::new(self.screen_rect.size.width as u32,
                                       self.screen_rect.size.height as u32),
            layer_ubo: layer_ubo,
            //clips: mem::replace(&mut self.clips, Vec::new()),
            tiles: tiles,
            text_buffer: text_buffer,
            debug_rects: debug_rects,
            color_texture_id: self.color_texture_id,
            mask_texture_id: self.mask_texture_id,
        }
    }
}

fn compute_box_shadow_rect(box_bounds: &Rect<f32>,
                           box_offset: &Point2D<f32>,
                           spread_radius: f32)
                           -> Rect<f32> {
    let mut rect = (*box_bounds).clone();
    rect.origin.x += box_offset.x;
    rect.origin.y += box_offset.y;
    rect.inflate(spread_radius, spread_radius)
}

// FIXME(pcwalton): Assumes rectangles are well-formed with origin in TL
fn add_box_shadow_corner(top_left: &Point2D<f32>,
                         bottom_right: &Point2D<f32>,
                         corner_area_top_left: &Point2D<f32>,
                         corner_area_bottom_right: &Point2D<f32>,
                         color: &ColorF,
                         rotation_angle: RotationKind,
                         uv_rect: &RectUv<f32>,
                         is_opaque: bool,
                         part_list: &mut Vec<PrimitivePart>) {
    let rect = Rect::new(*top_left, Size2D::new(bottom_right.x - top_left.x,
                                                bottom_right.y - top_left.y));

    let mut part = PrimitivePart::box_shadow_texture(rect,
                                                     *color,
                                                     uv_rect.top_left,
                                                     uv_rect.bottom_right,
                                                     is_opaque,
                                                     rotation_angle,
                                                     RectangleKind::Solid);

    let corner_area_rect =
        Rect::new(*corner_area_top_left,
                  Size2D::new(corner_area_bottom_right.x - corner_area_top_left.x,
                              corner_area_bottom_right.y - corner_area_top_left.y));
    let clip = Clip::from_rect(&corner_area_rect, ClipKind::ClipOut);

    // TODO(gw): This can only ever return 1 or none - make a fast path without heap allocs!
    let parts = part.clip(&clip);
    for part in parts {
        part_list.push(part);
    }
}

fn add_box_shadow_edge(top_left: &Point2D<f32>,
                       bottom_right: &Point2D<f32>,
                       color: &ColorF,
                       rotation_angle: RotationKind,
                       uv_rect: &RectUv<f32>,
                       is_opaque: bool,
                       part_list: &mut Vec<PrimitivePart>) {
    if top_left.x >= bottom_right.x || top_left.y >= bottom_right.y {
        return
    }

    let rect = Rect::new(*top_left, Size2D::new(bottom_right.x - top_left.x,
                                                bottom_right.y - top_left.y));

    part_list.push(PrimitivePart::box_shadow_texture(rect,
                                                     *color,
                                                     uv_rect.top_left,
                                                     uv_rect.bottom_right,
                                                     is_opaque,
                                                     rotation_angle,
                                                     RectangleKind::BoxShadowEdge));
}

fn add_box_shadow_sides(metrics: &BoxShadowMetrics,
                        color: &ColorF,
                        uv_rect: &RectUv<f32>,
                        is_opaque: bool,
                        part_list: &mut Vec<PrimitivePart>) {
    // Draw the sides.
    //
    //      +--+------------------+--+
    //      |  |##################|  |
    //      +--+------------------+--+
    //      |##|                  |##|
    //      |##|                  |##|
    //      |##|                  |##|
    //      +--+------------------+--+
    //      |  |##################|  |
    //      +--+------------------+--+

    let horizontal_size = Size2D::new(metrics.br_inner.x - metrics.tl_inner.x,
                                      metrics.edge_size);
    let vertical_size = Size2D::new(metrics.edge_size,
                                    metrics.br_inner.y - metrics.tl_inner.y);
    let top_rect = Rect::new(metrics.tl_outer + Point2D::new(metrics.edge_size, 0.0),
                             horizontal_size);
    let right_rect =
        Rect::new(metrics.tr_outer + Point2D::new(-metrics.edge_size, metrics.edge_size),
                  vertical_size);
    let bottom_rect =
        Rect::new(metrics.bl_outer + Point2D::new(metrics.edge_size, -metrics.edge_size),
                  horizontal_size);
    let left_rect = Rect::new(metrics.tl_outer + Point2D::new(0.0, metrics.edge_size),
                              vertical_size);

    add_box_shadow_edge(&top_rect.origin,
                        &top_rect.bottom_right(),
                        color,
                        RotationKind::Angle90,
                        uv_rect,
                        is_opaque,
                        part_list);
    add_box_shadow_edge(&right_rect.origin,
                        &right_rect.bottom_right(),
                        color,
                        RotationKind::Angle180,
                        uv_rect,
                        is_opaque,
                        part_list);
    add_box_shadow_edge(&bottom_rect.origin,
                        &bottom_rect.bottom_right(),
                        color,
                        RotationKind::Angle270,
                        uv_rect,
                        is_opaque,
                        part_list);
    add_box_shadow_edge(&left_rect.origin,
                        &left_rect.bottom_right(),
                        color,
                        RotationKind::Angle0,
                        uv_rect,
                        is_opaque,
                        part_list);
}

fn add_box_shadow_corners(metrics: &BoxShadowMetrics,
                          box_bounds: &Rect<f32>,
                          color: &ColorF,
                          uv_rect: &RectUv<f32>,
                          is_opaque: bool,
                          part_list: &mut Vec<PrimitivePart>) {
    // Draw the corners.
    //
    //      +--+------------------+--+
    //      |##|                  |##|
    //      +--+------------------+--+
    //      |  |                  |  |
    //      |  |                  |  |
    //      |  |                  |  |
    //      +--+------------------+--+
    //      |##|                  |##|
    //      +--+------------------+--+

    // Prevent overlap of the box shadow corners when the size of the blur is larger than the
    // size of the box.
    let center = Point2D::new(box_bounds.origin.x + box_bounds.size.width / 2.0,
                              box_bounds.origin.y + box_bounds.size.height / 2.0);

    add_box_shadow_corner(&metrics.tl_outer,
                          &Point2D::new(metrics.tl_outer.x + metrics.edge_size,
                                        metrics.tl_outer.y + metrics.edge_size),
                          &metrics.tl_outer,
                          &center,
                          &color,
                          RotationKind::Angle0,
                          uv_rect,
                          is_opaque,
                          part_list);
    add_box_shadow_corner(&Point2D::new(metrics.tr_outer.x - metrics.edge_size,
                                        metrics.tr_outer.y),
                          &Point2D::new(metrics.tr_outer.x,
                                        metrics.tr_outer.y + metrics.edge_size),
                          &Point2D::new(center.x, metrics.tr_outer.y),
                          &Point2D::new(metrics.tr_outer.x, center.y),
                          &color,
                          RotationKind::Angle90,
                          uv_rect,
                          is_opaque,
                          part_list);
    add_box_shadow_corner(&Point2D::new(metrics.br_outer.x - metrics.edge_size,
                                         metrics.br_outer.y - metrics.edge_size),
                          &Point2D::new(metrics.br_outer.x, metrics.br_outer.y),
                          &center,
                          &metrics.br_outer,
                          &color,
                          RotationKind::Angle180,
                          uv_rect,
                          is_opaque,
                          part_list);
    add_box_shadow_corner(&Point2D::new(metrics.bl_outer.x,
                                        metrics.bl_outer.y - metrics.edge_size),
                          &Point2D::new(metrics.bl_outer.x + metrics.edge_size,
                                        metrics.bl_outer.y),
                          &Point2D::new(metrics.bl_outer.x, center.y),
                          &Point2D::new(center.x, metrics.bl_outer.y),
                          &color,
                          RotationKind::Angle270,
                          uv_rect,
                          is_opaque,
                          part_list);
}
