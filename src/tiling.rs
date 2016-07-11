/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::{Au};
use batch_builder::{BorderSideHelpers, BoxShadowMetrics};
use bsptree::BspTree;
use device::{TextureId, TextureFilter};
use euclid::{Point2D, Rect, Matrix4D, Size2D, Point4D};
use fnv::FnvHasher;
use frame::FrameId;
use internal_types::{AxisDirection, Glyph, GlyphKey, DevicePixel};
use layer::Layer;
use renderer::{BLUR_INFLATION_FACTOR};
use resource_cache::ResourceCache;
use resource_list::ResourceList;
use std::cmp;
use std::collections::{HashMap};
use std::f32;
use std::mem;
use std::hash::{BuildHasherDefault};
use texture_cache::{TexturePage, TextureCacheItem};
use util::{self, rect_from_points, rect_from_points_f, MatrixHelpers, subtract_rect, RectHelpers, rect_contains_rect};
use webrender_traits::{ColorF, FontKey, ImageKey, ImageRendering, ComplexClipRegion};
use webrender_traits::{BorderDisplayItem, BorderStyle, ItemRange, AuxiliaryLists, BorderRadius};
use webrender_traits::{BoxShadowClipMode, PipelineId, ScrollLayerId};

pub struct RenderTarget {
    pub index: usize,
    page_allocator: TexturePage,
    prim_lists: Vec<PackedPrimList>,

    pub cache_batches: Vec<PrimitiveBatch>,

    pub layer_ubo: Vec<PackedLayer>,
    pub layer_to_ubo_map: Vec<Option<usize>>,

    pub tile_ubo: Vec<PackedTile>,

    pub composite_batches: HashMap<CompositeBatchKey,
                                   Vec<CompositeTile>,
                                   BuildHasherDefault<FnvHasher>>,
}

impl RenderTarget {
    fn new(index: usize, max_layers: usize) -> RenderTarget {
        let mut layer_to_ubo_map = Vec::with_capacity(max_layers);

        for _ in 0..max_layers {
            layer_to_ubo_map.push(None);
        }

        RenderTarget {
            index: index,
            page_allocator: TexturePage::new(TextureId(0), RENDERABLE_CACHE_SIZE.0 as u32),
            prim_lists: Vec::new(),
            layer_to_ubo_map: layer_to_ubo_map,
            layer_ubo: Vec::new(),
            cache_batches: Vec::new(),
            tile_ubo: Vec::new(),
            composite_batches: HashMap::with_hasher(Default::default()),
        }
    }

    fn build(&mut self) {
        // Build batches
        loop {
            // Pull next primitive
            let mut batch = None;

            for prim_list in &mut self.prim_lists {
                if let Some(next_prim) = prim_list.primitives.pop() {
                    let mut new_batch = PrimitiveBatch::new(&next_prim);
                    new_batch.push(&next_prim, prim_list.color_texture_id);
                    batch = Some(new_batch);
                    break;
                }
            }

            match batch {
                Some(mut batch) => {
                    for prim_list in &mut self.prim_lists {
                        loop {
                            match prim_list.primitives.pop() {
                                Some(next_prim) => {
                                    if !batch.push(&next_prim, prim_list.color_texture_id) {
                                        prim_list.primitives.push(next_prim);
                                        break;
                                    }
                                }
                                None => {
                                    break;
                                }
                            }
                        }
                    }

                    self.cache_batches.push(batch);
                }
                None => {
                    break;
                }
            }
        }
    }
}

pub struct RenderPhase {
    pub targets: Vec<RenderTarget>,
    max_layers: usize,
}

impl RenderPhase {
    fn new(max_layers: usize) -> RenderPhase {
        let main_target = RenderTarget::new(0, max_layers);

        RenderPhase {
            targets: vec![main_target],
            max_layers: max_layers,
        }
    }

    fn alloc_render_rect(&mut self,
                         target_index: usize,
                         size: &Size2D<DevicePixel>) -> Option<Point2D<DevicePixel>> {
        debug_assert!(target_index > 0);
        if target_index == self.targets.len() {
            let index = self.targets.len();
            self.targets.push(RenderTarget::new(index, self.max_layers));
        }
        let target = &mut self.targets[target_index];

        let alloc_size = Size2D::new(size.width.0 as u32, size.height.0 as u32);

        let origin = target.page_allocator
                           .allocate(&alloc_size, TextureFilter::Linear);

        origin.map(|o| {
            Point2D::new(DevicePixel(o.x as i32), DevicePixel(o.y as i32))
        })
    }

    fn add_layer_to_ubo_if_needed(&mut self,
                                  target_index: usize,
                                  layer_index: StackingContextIndex,
                                  layer: &StackingContext) -> u32 {
        let target = &mut self.targets[target_index];

        let StackingContextIndex(si) = layer_index;

        let layer_index_in_ubo = match target.layer_to_ubo_map[si] {
            Some(index) => index,
            None => {
                let index = target.layer_ubo.len();
                target.layer_ubo.push(PackedLayer {
                    padding: [0, 0],
                    transform: layer.transform,
                    inv_transform: layer.transform.invert(),
                    screen_vertices: layer.xf_rect.as_ref().unwrap().vertices,
                    blend_info: [layer.opacity, 0.0],
                });
                target.layer_to_ubo_map[si] = Some(index);
                index
            }
        };

        layer_index_in_ubo as u32
    }

    fn add_tile(&mut self,
                target_index: usize,
                actual_rect: &Rect<DevicePixel>,
                target_rect: &Rect<DevicePixel>) -> u32 {
        let tile_index_in_ubo = self.targets[target_index].tile_ubo.len();
        self.targets[target_index].tile_ubo.push(PackedTile {
            actual_rect: *actual_rect,
            target_rect: *target_rect,
        });
        tile_index_in_ubo as u32
    }

    fn add_prim_list(&mut self,
                     target_index: usize,
                     prim_list: PackedPrimList) {
        self.targets[target_index].prim_lists.push(prim_list)
    }

    fn add_composite(&mut self,
                     target_index: usize,
                     key: CompositeBatchKey,
                     composite_tile: CompositeTile) {
        let target = &mut self.targets[target_index];

        let batch = target.composite_batches.entry(key).or_insert_with(|| {
            Vec::new()
        });
        batch.push(composite_tile);
    }

    fn build(&mut self) {
        for target in &mut self.targets {
            target.build();
        }
    }
}

pub const SCREEN_TILE_SIZE: usize = 64;
pub const RENDERABLE_CACHE_SIZE: DevicePixel = DevicePixel(2048);
pub const MAX_LAYERS_PER_PASS: usize = 8;

#[allow(non_camel_case_types)]
#[derive(Debug, Eq, PartialEq, Copy, Clone, Hash)]
pub enum CompositeShader {
    Prim1,
    Prim2,
    Prim3,
    Prim4,
    Prim5,
    Prim6,
    Prim7,
    Prim8,
    //Partial1,
    //Partial2,
    /*
    Partial3,
    Partial4,
    Partial5,
    Partial6,
    Partial7,
    Partial8,*/
}

#[derive(Debug, Clone)]
pub struct DebugRect {
    pub label: String,
    pub color: ColorF,
    pub rect: Rect<DevicePixel>,
}

#[derive(Debug, Copy, Clone)]
enum CacheSize {
    None,
    Fixed,
    Variable(Size2D<DevicePixel>),
}

#[derive(Debug, Clone)]
enum TransformedRectKind {
    AxisAligned,
    Complex,
}

#[derive(Debug, Clone)]
struct TransformedRect {
    local_rect: Rect<f32>,
    bounding_rect: Rect<DevicePixel>,
    vertices: [Point4D<f32>; 4],
    kind: TransformedRectKind,
}

impl TransformedRect {
    fn new(rect: &Rect<f32>,
           transform: &Matrix4D<f32>,
           device_pixel_ratio: f32) -> TransformedRect {

        let kind = if transform.can_losslessly_transform_and_perspective_project_a_2d_rect() {
            TransformedRectKind::AxisAligned
        } else {
            TransformedRectKind::Complex
        };

        match kind {
            TransformedRectKind::AxisAligned => {
                let v0 = transform.transform_point(&rect.origin);
                let v1 = transform.transform_point(&rect.top_right());
                let v2 = transform.transform_point(&rect.bottom_left());
                let v3 = transform.transform_point(&rect.bottom_right());

                let screen_min_dp = Point2D::new(DevicePixel((v0.x * device_pixel_ratio).floor() as i32),
                                                 DevicePixel((v0.y * device_pixel_ratio).floor() as i32));
                let screen_max_dp = Point2D::new(DevicePixel((v3.x * device_pixel_ratio).ceil() as i32),
                                                 DevicePixel((v3.y * device_pixel_ratio).ceil() as i32));

                let screen_rect_dp = Rect::new(screen_min_dp, Size2D::new(screen_max_dp.x - screen_min_dp.x,
                                                                          screen_max_dp.y - screen_min_dp.y));

                TransformedRect {
                    local_rect: *rect,
                    vertices: [
                        Point4D::new(v0.x, v0.y, 0.0, 1.0),
                        Point4D::new(v1.x, v1.y, 0.0, 1.0),
                        Point4D::new(v2.x, v2.y, 0.0, 1.0),
                        Point4D::new(v3.x, v3.y, 0.0, 1.0),
                    ],
                    bounding_rect: screen_rect_dp,
                    kind: kind,
                }
            }
            TransformedRectKind::Complex => {
                println!("complex!");

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

                let mut screen_min: Point2D<f32> = Point2D::new( 10000000.0,  10000000.0);
                let mut screen_max: Point2D<f32> = Point2D::new(-10000000.0, -10000000.0);

                for vertex in &vertices {
                    let inv_w = 1.0 / vertex.w;
                    let vx = vertex.x * inv_w;
                    let vy = vertex.y * inv_w;
                    screen_min.x = screen_min.x.min(vx);
                    screen_min.y = screen_min.y.min(vy);
                    screen_max.x = screen_max.x.max(vx);
                    screen_max.y = screen_max.y.max(vy);
                }

                let screen_min_dp = Point2D::new(DevicePixel((screen_min.x * device_pixel_ratio).floor() as i32),
                                                 DevicePixel((screen_min.y * device_pixel_ratio).floor() as i32));
                let screen_max_dp = Point2D::new(DevicePixel((screen_max.x * device_pixel_ratio).ceil() as i32),
                                                 DevicePixel((screen_max.y * device_pixel_ratio).ceil() as i32));

                let screen_rect_dp = Rect::new(screen_min_dp, Size2D::new(screen_max_dp.x - screen_min_dp.x,
                                                                          screen_max_dp.y - screen_min_dp.y));

                TransformedRect {
                    local_rect: *rect,
                    vertices: vertices,
                    bounding_rect: screen_rect_dp,
                    kind: kind,
                }
            }
        }
    }
}

#[derive(Debug)]
struct RectanglePrimitive {
    color: ColorF,
}

#[derive(Debug)]
struct TextPrimitive {
    color: ColorF,
    font_key: FontKey,
    size: Au,
    blur_radius: Au,
    glyph_range: ItemRange,
}

#[derive(Debug)]
struct BoxShadowPrimitive {
    src_rect: Rect<f32>,
    bs_rect: Rect<f32>,
    color: ColorF,
    blur_radius: f32,
    spread_radius: f32,
    border_radius: f32,
    clip_mode: BoxShadowClipMode,
    metrics: BoxShadowMetrics,
}

#[derive(Debug)]
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

#[derive(Debug)]
struct ImagePrimitive {
    image_key: ImageKey,
    image_rendering: ImageRendering,
}

#[derive(Debug)]
struct GradientPrimitive {
    stops_range: ItemRange,
    dir: AxisDirection,
}

#[derive(Debug)]
enum PrimitiveDetails {
    Rectangle(RectanglePrimitive),
    Text(TextPrimitive),
    Image(ImagePrimitive),
    Border(BorderPrimitive),
    Gradient(GradientPrimitive),
    BoxShadow(BoxShadowPrimitive),
}

#[derive(Clone)]
enum PackedPrimitive {
    Rectangle(PackedRectanglePrimitive),
    Glyph(PackedGlyphPrimitive),
    Image(PackedImagePrimitive),
    Border(PackedBorderPrimitive),
    BoxShadow(PackedBoxShadowPrimitive),
}

struct PackedPrimList {
    color_texture_id: TextureId,
    primitives: Vec<PackedPrimitive>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PrimitiveIndex(usize);

#[derive(Debug)]
struct Primitive {
    rect: Rect<f32>,
    clip_index: Option<ClipIndex>,
    xf_rect: Option<TransformedRect>,
    details: PrimitiveDetails,
}

impl Primitive {
    fn is_opaque(&self) -> bool {
        match self.details {
            PrimitiveDetails::Rectangle(ref details) => {
                details.color.a == 1.0
            }
            _ => {
                false
            }
        }
    }

    fn pack(&self,
            prim_list: &mut PackedPrimList,
            layer_index_in_ubo: u32,
            tile_index_in_ubo: u32,
            auxiliary_lists: &AuxiliaryLists,
            resource_cache: &ResourceCache,
            frame_id: FrameId,
            device_pixel_ratio: f32) {
        match self.details {
            PrimitiveDetails::Rectangle(ref details) => {
                prim_list.primitives.push(PackedPrimitive::Rectangle(PackedRectanglePrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        tile_index: tile_index_in_ubo,
                        layer_index: layer_index_in_ubo,
                        part: PrimitivePart::Invalid,
                    },
                    local_rect: self.rect,
                    color: details.color,
                }));
            }
            PrimitiveDetails::Image(ref details) => {
                let image_info = resource_cache.get_image(details.image_key,
                                                          details.image_rendering,
                                                          frame_id);
                let uv_rect = image_info.uv_rect();

                // TODO(gw): Need a general solution to handle multiple texture pages per tile in WR2!
                assert!(prim_list.color_texture_id == TextureId(0) ||
                        prim_list.color_texture_id == image_info.texture_id);
                prim_list.color_texture_id = image_info.texture_id;

                prim_list.primitives.push(PackedPrimitive::Image(PackedImagePrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        tile_index: tile_index_in_ubo,
                        layer_index: layer_index_in_ubo,
                        part: PrimitivePart::Invalid,
                    },
                    local_rect: self.rect,
                    st0: uv_rect.top_left,
                    st1: uv_rect.bottom_right,
                }));
            }
            PrimitiveDetails::Border(ref details) => {
                let inner_radius = BorderRadius {
                    top_left: Size2D::new(details.radius.top_left.width - details.left_width,
                                          details.radius.top_left.width - details.left_width),
                    top_right: Size2D::new(details.radius.top_right.width - details.right_width,
                                           details.radius.top_right.width - details.right_width),
                    bottom_left: Size2D::new(details.radius.bottom_left.width - details.left_width,
                                             details.radius.bottom_left.width - details.left_width),
                    bottom_right: Size2D::new(details.radius.bottom_right.width - details.right_width,
                                              details.radius.bottom_right.width - details.right_width),
                };

                let clip = Clip::from_border_radius(&self.rect,
                                                    &details.radius,
                                                    &inner_radius);

                prim_list.primitives.push(PackedPrimitive::Border(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        tile_index: tile_index_in_ubo,
                        layer_index: layer_index_in_ubo,
                        part: PrimitivePart::TopLeft,
                    },
                    local_rect: rect_from_points_f(details.tl_outer.x,
                                                   details.tl_outer.y,
                                                   details.tl_inner.x,
                                                   details.tl_inner.y),
                    color0: details.top_color,
                    color1: details.left_color,
                    outer_radius_x: details.radius.top_left.width,
                    outer_radius_y: details.radius.top_left.height,
                    inner_radius_x: inner_radius.top_left.width,
                    inner_radius_y: inner_radius.top_left.height,
                }));

                prim_list.primitives.push(PackedPrimitive::Border(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        tile_index: tile_index_in_ubo,
                        layer_index: layer_index_in_ubo,
                        part: PrimitivePart::TopRight,
                    },
                    local_rect: rect_from_points_f(details.tr_inner.x,
                                                   details.tr_outer.y,
                                                   details.tr_outer.x,
                                                   details.tr_inner.y),
                    color0: details.right_color,
                    color1: details.top_color,
                    outer_radius_x: details.radius.top_right.width,
                    outer_radius_y: details.radius.top_right.height,
                    inner_radius_x: inner_radius.top_right.width,
                    inner_radius_y: inner_radius.top_right.height,
                }));

                prim_list.primitives.push(PackedPrimitive::Border(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        tile_index: tile_index_in_ubo,
                        layer_index: layer_index_in_ubo,
                        part: PrimitivePart::BottomLeft,
                    },
                    local_rect: rect_from_points_f(details.bl_outer.x,
                                                   details.bl_inner.y,
                                                   details.bl_inner.x,
                                                   details.bl_outer.y),
                    color0: details.left_color,
                    color1: details.bottom_color,
                    outer_radius_x: details.radius.bottom_left.width,
                    outer_radius_y: details.radius.bottom_left.height,
                    inner_radius_x: inner_radius.bottom_left.width,
                    inner_radius_y: inner_radius.bottom_left.height,
                }));

                prim_list.primitives.push(PackedPrimitive::Border(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        tile_index: tile_index_in_ubo,
                        layer_index: layer_index_in_ubo,
                        part: PrimitivePart::BottomRight,
                    },
                    local_rect: rect_from_points_f(details.br_inner.x,
                                                   details.br_inner.y,
                                                   details.br_outer.x,
                                                   details.br_outer.y),
                    color0: details.right_color,
                    color1: details.bottom_color,
                    outer_radius_x: details.radius.bottom_right.width,
                    outer_radius_y: details.radius.bottom_right.height,
                    inner_radius_x: inner_radius.bottom_right.width,
                    inner_radius_y: inner_radius.bottom_right.height,
                }));

                prim_list.primitives.push(PackedPrimitive::Border(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        tile_index: tile_index_in_ubo,
                        layer_index: layer_index_in_ubo,
                        part: PrimitivePart::Left,
                    },
                    local_rect: rect_from_points_f(details.tl_outer.x,
                                                   details.tl_inner.y,
                                                   details.tl_outer.x + details.left_width,
                                                   details.bl_inner.y),
                    color0: details.left_color,
                    color1: details.left_color,
                    outer_radius_x: 0.0,
                    outer_radius_y: 0.0,
                    inner_radius_x: 0.0,
                    inner_radius_y: 0.0,
                }));

                prim_list.primitives.push(PackedPrimitive::Border(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        tile_index: tile_index_in_ubo,
                        layer_index: layer_index_in_ubo,
                        part: PrimitivePart::Right,
                    },
                    local_rect: rect_from_points_f(details.tr_outer.x - details.right_width,
                                                   details.tr_inner.y,
                                                   details.br_outer.x,
                                                   details.br_inner.y),
                    color0: details.right_color,
                    color1: details.right_color,
                    outer_radius_x: 0.0,
                    outer_radius_y: 0.0,
                    inner_radius_x: 0.0,
                    inner_radius_y: 0.0,
                }));

                prim_list.primitives.push(PackedPrimitive::Border(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        tile_index: tile_index_in_ubo,
                        layer_index: layer_index_in_ubo,
                        part: PrimitivePart::Top,
                    },
                    local_rect: rect_from_points_f(details.tl_inner.x,
                                                   details.tl_outer.y,
                                                   details.tr_inner.x,
                                                   details.tr_outer.y + details.top_width),
                    color0: details.top_color,
                    color1: details.top_color,
                    outer_radius_x: 0.0,
                    outer_radius_y: 0.0,
                    inner_radius_x: 0.0,
                    inner_radius_y: 0.0,
                }));

                prim_list.primitives.push(PackedPrimitive::Border(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        tile_index: tile_index_in_ubo,
                        layer_index: layer_index_in_ubo,
                        part: PrimitivePart::Bottom,
                    },
                    local_rect: rect_from_points_f(details.bl_inner.x,
                                                   details.bl_outer.y - details.bottom_width,
                                                   details.br_inner.x,
                                                   details.br_outer.y),
                    color0: details.bottom_color,
                    color1: details.bottom_color,
                    outer_radius_x: 0.0,
                    outer_radius_y: 0.0,
                    inner_radius_x: 0.0,
                    inner_radius_y: 0.0,
                }));
            }
            PrimitiveDetails::Gradient(ref details) => {
                //println!("todo - gradient");
                /*
                let stops = auxiliary_lists.gradient_stops(&details.stops_range);
                for i in 0..(stops.len() - 1) {
                    let (prev_stop, next_stop) = (&stops[i], &stops[i + 1]);
                    let piece_origin;
                    let piece_size;
                    match details.dir {
                        AxisDirection::Horizontal => {
                            let prev_x = util::lerp(self.rect.origin.x, self.rect.max_x(), prev_stop.offset);
                            let next_x = util::lerp(self.rect.origin.x, self.rect.max_x(), next_stop.offset);
                            piece_origin = Point2D::new(prev_x, self.rect.origin.y);
                            piece_size = Size2D::new(next_x - prev_x, self.rect.size.height);
                        }
                        AxisDirection::Vertical => {
                            let prev_y = util::lerp(self.rect.origin.y, self.rect.max_y(), prev_stop.offset);
                            let next_y = util::lerp(self.rect.origin.y, self.rect.max_y(), next_stop.offset);
                            piece_origin = Point2D::new(self.rect.origin.x, prev_y);
                            piece_size = Size2D::new(self.rect.size.width, next_y - prev_y);
                        }
                    }

                    let piece_rect = Rect::new(piece_origin, piece_size);

                    batch_data.push(PackedGradientPrimitive {
                        common: PackedPrimitiveInfo {
                            padding: [0, 0],
                            part: PrimitivePart::Invalid,
                            renderable_index: renderable_index,
                        },
                        local_rect: piece_rect,
                        color0: prev_stop.color,
                        color1: next_stop.color,
                        padding: [0, 0, 0],
                        dir: details.dir,
                    });
                }
                */
            }
            PrimitiveDetails::BoxShadow(ref details) => {
                let mut rects = Vec::new();
                subtract_rect(&self.rect, &details.src_rect, &mut rects);

                for rect in rects {
                    prim_list.primitives.push(PackedPrimitive::BoxShadow(PackedBoxShadowPrimitive {
                        common: PackedPrimitiveInfo {
                            padding: 0,
                            tile_index: tile_index_in_ubo,
                            layer_index: layer_index_in_ubo,
                            part: PrimitivePart::Invalid,
                        },
                        local_rect: rect,
                        color: details.color,
                    }));
                }
            }
            PrimitiveDetails::Text(ref details) => {
                let src_glyphs = auxiliary_lists.glyph_instances(&details.glyph_range);
                let mut glyph_key = GlyphKey::new(details.font_key,
                                                  details.size,
                                                  details.blur_radius,
                                                  src_glyphs[0].index);
                let blur_offset = details.blur_radius.to_f32_px() * (BLUR_INFLATION_FACTOR as f32) / 2.0;

                for glyph in src_glyphs {
                    glyph_key.index = glyph.index;
                    let image_info = resource_cache.get_glyph(&glyph_key, frame_id);
                    if let Some(image_info) = image_info {
                        // TODO(gw): Need a general solution to handle multiple texture pages per tile in WR2!
                        assert!(prim_list.color_texture_id == TextureId(0) ||
                                prim_list.color_texture_id == image_info.texture_id);
                        prim_list.color_texture_id = image_info.texture_id;

                        let x = glyph.x + image_info.user_data.x0 as f32 / device_pixel_ratio - blur_offset;
                        let y = glyph.y - image_info.user_data.y0 as f32 / device_pixel_ratio - blur_offset;

                        let width = image_info.requested_rect.size.width as f32 / device_pixel_ratio;
                        let height = image_info.requested_rect.size.height as f32 / device_pixel_ratio;

                        let uv_rect = image_info.uv_rect();

                        prim_list.primitives.push(PackedPrimitive::Glyph(PackedGlyphPrimitive {
                            common: PackedPrimitiveInfo {
                                padding: 0,
                                tile_index: tile_index_in_ubo,
                                layer_index: layer_index_in_ubo,
                                part: PrimitivePart::Invalid,
                            },
                            local_rect: Rect::new(Point2D::new(x, y),
                                                  Size2D::new(width, height)),
                            color: details.color,
                            st0: uv_rect.top_left,
                            st1: uv_rect.bottom_right,
                        }));
                    }
                }
            }
        }
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone)]
enum PrimitivePart {
    Invalid = 0,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Top,
    Left,
    Bottom,
    Right,
}

#[derive(Debug, Clone)]
pub struct CompositeTile {
    pub screen_rect: Rect<DevicePixel>,
    pub src_rects: [Rect<DevicePixel>; MAX_LAYERS_PER_PASS],
    pub blend_info: [f32; MAX_LAYERS_PER_PASS],
}

impl CompositeTile {
    fn new(rect: &Rect<DevicePixel>) -> CompositeTile {
        CompositeTile {
            screen_rect: *rect,
            src_rects: unsafe { mem::uninitialized() },
            blend_info: [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ],
        }
    }
}

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct CompositeBatchKey {
    pub shader: CompositeShader,
    //pub samplers: [TextureId; MAX_PRIMS_PER_COMPOSITE],
}

impl CompositeBatchKey {
    fn new(shader: CompositeShader,
           //samplers: [TextureId; MAX_PRIMS_PER_COMPOSITE]
           ) -> CompositeBatchKey {
        CompositeBatchKey {
            shader: shader,
            //samplers: samplers,
        }
    }
}

#[derive(Debug)]
pub struct PackedTile {
    actual_rect: Rect<DevicePixel>,
    target_rect: Rect<DevicePixel>,
}

#[derive(Debug)]
pub struct PackedLayer {
    transform: Matrix4D<f32>,
    inv_transform: Matrix4D<f32>,
    screen_vertices: [Point4D<f32>; 4],
    blend_info: [f32; 2],
    padding: [u32; 2],
}

#[derive(Debug)]
pub struct PackedRenderable {
    transform: Matrix4D<f32>,
    local_rect: Rect<f32>,
    cache_rect: Rect<DevicePixel>,
    screen_rect: Rect<DevicePixel>,
    st0: Point2D<f32>,
    st1: Point2D<f32>,
    offset: Point2D<f32>,
    blend_info: [f32; 2],
}

#[derive(Debug, Clone)]
pub struct PackedPrimitiveInfo {
    layer_index: u32,
    tile_index: u32,
    part: PrimitivePart,
    padding: u32,
}

#[derive(Debug)]
pub struct PackedFixedRectangle {
    common: PackedPrimitiveInfo,
    color: ColorF,
}

#[derive(Debug, Clone)]
pub struct PackedRectanglePrimitive {
    common: PackedPrimitiveInfo,
    local_rect: Rect<f32>,
    color: ColorF,
}

#[derive(Debug, Clone)]
pub struct PackedGlyphPrimitive {
    common: PackedPrimitiveInfo,
    local_rect: Rect<f32>,
    color: ColorF,
    st0: Point2D<f32>,
    st1: Point2D<f32>,
}

#[derive(Debug, Clone)]
pub struct PackedImagePrimitive {
    common: PackedPrimitiveInfo,
    local_rect: Rect<f32>,
    st0: Point2D<f32>,
    st1: Point2D<f32>,
}

#[derive(Debug)]
pub struct PackedGradientPrimitive {
    common: PackedPrimitiveInfo,
    local_rect: Rect<f32>,
    color0: ColorF,
    color1: ColorF,
    dir: AxisDirection,
    padding: [u32; 3],
}

#[derive(Debug, Clone)]
pub struct PackedBorderPrimitive {
    common: PackedPrimitiveInfo,
    local_rect: Rect<f32>,
    color0: ColorF,
    color1: ColorF,
    outer_radius_x: f32,
    outer_radius_y: f32,
    inner_radius_x: f32,
    inner_radius_y: f32,
}

#[derive(Debug, Clone)]
pub struct PackedBoxShadowPrimitive {
    common: PackedPrimitiveInfo,
    local_rect: Rect<f32>,
    color: ColorF,
}

#[derive(Debug)]
pub enum PrimitiveBatchData {
    Rectangles(Vec<PackedRectanglePrimitive>),
    Borders(Vec<PackedBorderPrimitive>),
    BoxShadows(Vec<PackedBoxShadowPrimitive>),
    Text(Vec<PackedGlyphPrimitive>),
    Image(Vec<PackedImagePrimitive>),
    Gradient(Vec<PackedGradientPrimitive>),
}

#[derive(Debug)]
pub struct PrimitiveBatch {
    pub color_texture_id: TextureId,        // TODO(gw): Expand to sampler array to handle all glyphs!
    pub data: PrimitiveBatchData,
}

impl PrimitiveBatch {
    fn new(prim: &PackedPrimitive) -> PrimitiveBatch {
        let data = match prim {
            &PackedPrimitive::Rectangle(..) => {
                PrimitiveBatchData::Rectangles(Vec::new())
            }
            &PackedPrimitive::Border(..) => {
                PrimitiveBatchData::Borders(Vec::new())
            }
            &PackedPrimitive::BoxShadow(..) => {
                PrimitiveBatchData::BoxShadows(Vec::new())
            }
            &PackedPrimitive::Glyph(..) => {
                PrimitiveBatchData::Text(Vec::new())
            }
            &PackedPrimitive::Image(..) => {
                PrimitiveBatchData::Image(Vec::new())
            }
            /*
            PrimitiveDetails::Gradient(..) => {
                PrimitiveBatchData::Gradient(Vec::new())
            }*/
        };

        let mut this = PrimitiveBatch {
            color_texture_id: TextureId(0),
            data: data,
        };

        this
    }

    fn push(&mut self,
            prim: &PackedPrimitive,
            color_texture_id: TextureId) -> bool {
        if color_texture_id != TextureId(0) {
            assert!(self.color_texture_id == TextureId(0) ||
                    self.color_texture_id == color_texture_id);
            self.color_texture_id = color_texture_id;
        }

        match (&mut self.data, prim) {
            (&mut PrimitiveBatchData::Rectangles(ref mut data), &PackedPrimitive::Rectangle(ref prim)) => {
                data.push(prim.clone());        // fixme!!!!!!!!!!!!!
                true
            }
            (_, &PackedPrimitive::Rectangle(ref prim)) => {
                false
            }
            (&mut PrimitiveBatchData::Text(ref mut data), &PackedPrimitive::Glyph(ref prim)) => {
                data.push(prim.clone());        // fixme!!!!!!!!!!!!!
                true
            }
            (_, &PackedPrimitive::Glyph(ref prim)) => {
                false
            }
            (&mut PrimitiveBatchData::Image(ref mut data), &PackedPrimitive::Image(ref prim)) => {
                data.push(prim.clone());        // fixme!!!!!!!!!!!!!
                true
            }
            (_, &PackedPrimitive::Image(ref prim)) => {
                false
            }
            (&mut PrimitiveBatchData::Borders(ref mut data), &PackedPrimitive::Border(ref prim)) => {
                data.push(prim.clone());        // fixme!!!!!!!!!!!!!
                true
            }
            (_, &PackedPrimitive::Border(ref prim)) => {
                false
            }
            (&mut PrimitiveBatchData::BoxShadows(ref mut data), &PackedPrimitive::BoxShadow(ref prim)) => {
                data.push(prim.clone());        // fixme!!!!!!!!!!!!!
                true
            }
            (_, &PackedPrimitive::BoxShadow(ref prim)) => {
                false
            }
            //_ => panic!("todo"),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct RenderableIndex(usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct StackingContextChunkIndex(usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct StackingContextIndex(usize);

struct StackingContext {
    pipeline_id: PipelineId,
    local_transform: Matrix4D<f32>,
    local_rect: Rect<f32>,
    local_offset: Point2D<f32>,
    primitives: Vec<Primitive>,
    scroll_layer_id: ScrollLayerId,
    opacity: f32,
    transform: Matrix4D<f32>,
    xf_rect: Option<TransformedRect>,
}

impl StackingContext {
    fn build_resource_list(&self,
                           resource_list: &mut ResourceList,
                           //index_buffer: &Vec<PrimitiveIndex>,
                           auxiliary_lists: &AuxiliaryLists) {
//        for prim_index in index_buffer {
//            let PrimitiveIndex(prim_index) = *prim_index;
//            let prim = &self.primitives[prim_index];
        for prim in &self.primitives {
            match prim.details {
                PrimitiveDetails::Rectangle(..) => {}
                PrimitiveDetails::Gradient(..) => {}
                PrimitiveDetails::Border(..) => {}
                PrimitiveDetails::BoxShadow(..) => {}
                PrimitiveDetails::Image(ref details) => {
                   resource_list.add_image(details.image_key,
                                            details.image_rendering);
                }
                PrimitiveDetails::Text(ref details) => {
                    let glyphs = auxiliary_lists.glyph_instances(&details.glyph_range);
                    for glyph in glyphs {
                        let glyph = Glyph::new(details.size, details.blur_radius, glyph.index);
                        resource_list.add_glyph(details.font_key, glyph);
                    }
                }
            }
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct ClipIndex(usize);

#[derive(Debug, Clone)]
pub struct ClipCorner {
    rect: Rect<f32>,
    outer_radius_x: f32,
    outer_radius_y: f32,
    inner_radius_x: f32,
    inner_radius_y: f32,
}

#[derive(Debug, Clone)]
pub struct Clip {
    rect: Rect<f32>,
    top_left: ClipCorner,
    top_right: ClipCorner,
    bottom_left: ClipCorner,
    bottom_right: ClipCorner,
}

#[derive(Debug, Clone)]
pub struct ErrorTile {
    pub rect: Rect<DevicePixel>,
}

#[derive(Debug, Clone)]
pub struct ClearTile {
    pub rect: Rect<DevicePixel>,
}

pub struct FrameBuilderConfig {

}

impl FrameBuilderConfig {
    pub fn new() -> FrameBuilderConfig {
        FrameBuilderConfig {

        }
    }
}

pub struct FrameBuilder {
    screen_rect: Rect<i32>,
    layers: Vec<StackingContext>,
    layer_stack: Vec<StackingContextIndex>,
    clips: Vec<Clip>,
    clip_stack: Vec<ClipIndex>,
    device_pixel_ratio: f32,
    debug: bool,
}

pub struct Frame {
    pub debug_rects: Vec<DebugRect>,
    pub cache_size: Size2D<f32>,
    pub phases: Vec<RenderPhase>,
    pub clear_tiles: Vec<ClearTile>,
    pub error_tiles: Vec<ErrorTile>,
}

impl Clip {
    pub fn from_clip_region(clip: &ComplexClipRegion) -> Clip {
        Clip {
            rect: clip.rect,
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

    pub fn from_border_radius(rect: &Rect<f32>,
                              outer_radius: &BorderRadius,
                              inner_radius: &BorderRadius) -> Clip {
        Clip {
            rect: *rect,
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
}

#[derive(Copy, Clone, Debug)]
struct RenderableCacheEntry {
    rendered: bool,
    rect: Rect<DevicePixel>,
    st0: Point2D<f32>,
    st1: Point2D<f32>,
}

struct RenderableCache {
    entries: Vec<Option<RenderableCacheEntry>>,
    target_size: Size2D<f32>,
    page_allocator: TexturePage,
}

impl RenderableCache {
    fn new(renderable_count: usize,
           size: DevicePixel) -> RenderableCache {
        let target_size = Size2D::new(size.0 as f32, size.0 as f32);

        let mut entries = Vec::with_capacity(renderable_count);

        for i in 0..renderable_count {
            entries.push(None);
        }

        RenderableCache {
            entries: entries,
            page_allocator: TexturePage::new(TextureId(0), size.0 as u32),
            target_size: target_size,
        }
    }

    fn clear(&mut self) {
        self.page_allocator.clear();
        for entry in &mut self.entries {
            *entry = None;
        }
    }

    fn get(&mut self, key: RenderableIndex) -> RenderableCacheEntry {
        let RenderableIndex(ri) = key;
        let entry = self.entries[ri].as_mut().unwrap();
        let result = entry.clone();
        entry.rendered = true;
        result
    }

    fn allocate_if_needed(&mut self,
                          key: RenderableIndex,
                          cache_size: CacheSize) -> bool {
        let RenderableIndex(ri) = key;

        match self.entries[ri] {
            Some(entry) => {
                true
            }
            None => {
                let alloc_size = match cache_size {
                    CacheSize::Fixed => {
                        Size2D::new(2, 2)
                    }
                    CacheSize::Variable(size) => {
                        Size2D::new(size.width.0 as u32,
                                    size.height.0 as u32)
                    }
                    CacheSize::None => unreachable!(),
                };

                let origin = self.page_allocator
                                 .allocate(&alloc_size, TextureFilter::Linear);

                match origin {
                    Some(origin) => {
                        let (st0, st1) = match cache_size {
                            CacheSize::Fixed => {
                                let st = Point2D::new((origin.x + 1) as f32 / self.target_size.width,
                                                      (origin.y + 1) as f32 / self.target_size.height);

                                (st, st)
                            }
                            CacheSize::Variable(size) => {
                                let st0 = Point2D::new(origin.x as f32 / self.target_size.width,
                                                       origin.y as f32 / self.target_size.height);

                                let st1 = Point2D::new((origin.x + alloc_size.width) as f32 / self.target_size.width,
                                                       (origin.y + alloc_size.height) as f32 / self.target_size.height);

                                (st0, st1)
                            }
                            CacheSize::None => unreachable!(),
                        };

                        let entry = RenderableCacheEntry {
                            rendered: false,
                            rect: Rect::new(Point2D::new(DevicePixel(origin.x as i32),
                                                         DevicePixel(origin.y as i32)),
                                            Size2D::new(DevicePixel(alloc_size.width as i32),
                                                        DevicePixel(alloc_size.height as i32))),
                            st0: st0,
                            st1: st1,
                        };

                        self.entries[ri] = Some(entry);

                        true
                    }
                    None => {
                        false
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
struct ScreenTileLayer {
    layer_index: StackingContextIndex,
    prim_indices: Vec<PrimitiveIndex>,      // todo(gw): pre-build these into parts to save duplicated cpu time?
    layer_in_ubo: u32,
    layer_opacity: f32,
    is_opaque: bool,
}

impl ScreenTileLayer {
    fn compile(&mut self,
               layer: &StackingContext,
               screen_rect: &Rect<DevicePixel>) {
        self.prim_indices.sort_by(|a, b| {
            b.cmp(&a)
        });
        self.prim_indices.dedup();

/*
        // Intra-layer occlusion
        let first_opaque_cover_index = self.prim_indices.iter().position(|i| {
            let PrimitiveIndex(pi) = *i;
            let prim = &layer.primitives[pi];
            prim.is_opaque() &&
               rect_contains_rect(&prim.xf_rect.as_ref().unwrap().bounding_rect, screen_rect)
        });
        if let Some(first_opaque_cover_index) = first_opaque_cover_index {
            self.prim_indices.truncate(first_opaque_cover_index);
        }
*/

        // Inter-layer occlusion
        let PrimitiveIndex(pi) = *self.prim_indices.last().unwrap();
        let last_prim = &layer.primitives[pi];
        if self.layer_opacity == 1.0 &&
           last_prim.is_opaque() &&
           rect_contains_rect(&last_prim.xf_rect.as_ref().unwrap().bounding_rect,
                              screen_rect) {
            self.is_opaque = true;
        }
    }
}

#[derive(Debug)]
struct AllocationRequest {
    origin: Option<Point2D<DevicePixel>>,
    size: Size2D<DevicePixel>,
}

#[derive(Debug)]
struct TilePassResourceList {
    allocations: Vec<AllocationRequest>,
}

impl TilePassResourceList {
    fn new() -> TilePassResourceList {
        TilePassResourceList {
            allocations: Vec::new(),
        }
    }
}

#[derive(Debug)]
struct TileResourceList {
    passes: Vec<TilePassResourceList>,
}

impl TileResourceList {
    fn none() -> TileResourceList {
        TileResourceList {
            passes: Vec::new(),
        }
    }

    fn new() -> TileResourceList {
        TileResourceList {
            passes: Vec::new(),
        }
    }

    fn add_alloc(&mut self, pass_index: usize, size: &Size2D<DevicePixel>) {
        debug_assert!(pass_index <= self.passes.len());
        if pass_index == self.passes.len() {
            self.passes.push(TilePassResourceList::new());
        }
        self.passes[pass_index].allocations.push(AllocationRequest {
            origin: None,
            size: *size,
        });
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ScreenTileIndex(usize);

enum ScreenTileCompileResult {
    Clear,
    Unhandled,
    OutOfMemory,
    Ok,
}

#[derive(Debug)]
struct ScreenTile {
    rect: Rect<DevicePixel>,
    layers: Vec<ScreenTileLayer>,
    resource_list: Option<TileResourceList>,
}

impl ScreenTile {
    fn new(rect: Rect<DevicePixel>) -> ScreenTile {
        ScreenTile {
            rect: rect,
            layers: Vec::new(),
            resource_list: None,
        }
    }

    fn layer_count(&self) -> usize {
        self.layers.len()
    }

    fn add_layer_to_phase(&mut self,
                          target_index: usize,
                          target_rect: &Rect<DevicePixel>,
                          layer_index: usize,
                          layers: &Vec<StackingContext>,
                          pipeline_auxiliary_lists: &HashMap<PipelineId, AuxiliaryLists, BuildHasherDefault<FnvHasher>>,
                          resource_cache: &ResourceCache,
                          frame_id: FrameId,
                          device_pixel_ratio: f32,
                          phase: &mut RenderPhase) {
        let layer = &self.layers[layer_index];

        let StackingContextIndex(si) = layer.layer_index;
        let sc = &layers[si];

        let tile_index_in_ubo = phase.add_tile(target_index, &self.rect, &target_rect);
        let layer_index_in_ubo = phase.add_layer_to_ubo_if_needed(target_index, layer.layer_index, sc);

        let mut prim_list = PackedPrimList {
            primitives: Vec::new(),
            color_texture_id: TextureId(0),
        };

        let auxiliary_lists = pipeline_auxiliary_lists.get(&sc.pipeline_id)
                                                      .expect("No auxiliary lists?!");

        for prim_index in &layer.prim_indices {
            let PrimitiveIndex(pi) = *prim_index;
            let prim = &sc.primitives[pi];

            prim.pack(&mut prim_list,
                      layer_index_in_ubo as u32,
                      tile_index_in_ubo as u32,
                      auxiliary_lists,
                      resource_cache,
                      frame_id,
                      device_pixel_ratio)
        }

        phase.add_prim_list(target_index, prim_list);
    }

    fn build_resource_list(&mut self) {
        if self.layers.is_empty() {
            // Tile is empty - will be rendered by ps_clear
        } else if self.layers.len() == 1 {
            // Tile contains a single stacking context.
            // Render direct to frame buffer so no resources needed!
        } else if self.layers.len() <= MAX_LAYERS_PER_PASS {
            let mut resource_list = TileResourceList::new();
            for _ in 0..self.layers.len() {
                resource_list.add_alloc(0, &self.rect.size);
            }
            self.resource_list = Some(resource_list)
        } else {
            // Not currently supported - no resources needed
            // as ps_error will be used.
        }
    }

    fn compile(&mut self,
               layers: &Vec<StackingContext>,
               pipeline_auxiliary_lists: &HashMap<PipelineId, AuxiliaryLists, BuildHasherDefault<FnvHasher>>,
               resource_cache: &ResourceCache,
               frame_id: FrameId,
               device_pixel_ratio: f32,
               phase: &mut RenderPhase) -> ScreenTileCompileResult {
        // Allocate space in render targets as required.

        // Once that is done, should definitely be able to draw in
        // this phase (perhaps have to break batches for UBO placement etc)

        // Get all layers into the UBO for this phase

        if self.layers.is_empty() {
            ScreenTileCompileResult::Clear
        } else if self.layers.len() == 1 {
            let target_rect = self.rect;
            self.add_layer_to_phase(0,
                                    &target_rect,
                                    0,
                                    layers,
                                    pipeline_auxiliary_lists,
                                    resource_cache,
                                    frame_id,
                                    device_pixel_ratio,
                                    phase);

            ScreenTileCompileResult::Ok
        } else if self.layers.len() <= MAX_LAYERS_PER_PASS {
            let shader = match self.layers.len() {
                1 => CompositeShader::Prim1,
                2 => CompositeShader::Prim2,
                3 => CompositeShader::Prim3,
                4 => CompositeShader::Prim4,
                5 => CompositeShader::Prim5,
                6 => CompositeShader::Prim6,
                7 => CompositeShader::Prim7,
                8 => CompositeShader::Prim8,
                _ => panic!("todo - other shader?"),
            };

            let key = CompositeBatchKey::new(shader);
            let mut composite_tile = CompositeTile::new(&self.rect);

            for layer_index in 0..self.layers.len() {
                let p = phase.alloc_render_rect(1, &self.rect.size).expect("todo - alloc!");
                let r = Rect::new(p, self.rect.size);
                composite_tile.src_rects[layer_index] = r;
                composite_tile.blend_info[layer_index] = self.layers[layer_index].layer_opacity;

                self.add_layer_to_phase(1,
                                        &r,
                                        layer_index,
                                        layers,
                                        pipeline_auxiliary_lists,
                                        resource_cache,
                                        frame_id,
                                        device_pixel_ratio,
                                        phase);
            }

            phase.add_composite(0,
                                key,
                                composite_tile);

            ScreenTileCompileResult::Ok
        } else {
            ScreenTileCompileResult::Unhandled
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum RenderableStrategy {
    Unknown,
    AlphaBlendBatching,
    FixedRectangle,
    //PrimCache,
}

#[derive(Clone)]
struct Renderable {
    xf_rect: TransformedRect,
    layer_index: StackingContextIndex,
    index_buffer: Vec<PrimitiveIndex>,
    //cache_size: CacheSize,
    //strategy: RenderableStrategy,
    tile_hit_count: usize,
}

impl FrameBuilder {
    pub fn new(viewport_size: Size2D<f32>,
               device_pixel_ratio: f32,
               debug: bool) -> FrameBuilder {
        let viewport_size = Size2D::new(viewport_size.width as i32, viewport_size.height as i32);
        FrameBuilder {
            screen_rect: Rect::new(Point2D::zero(), viewport_size),
            layers: Vec::new(),
            layer_stack: Vec::new(),
            device_pixel_ratio: device_pixel_ratio,
            debug: debug,
            clips: Vec::new(),
            clip_stack: Vec::new(),
        }
    }

    pub fn set_clip(&mut self, clip: Clip) {
        let clip_index = ClipIndex(self.clips.len());
        // TODO(gw): Handle intersecting clips!
        self.clips.push(clip);
        self.clip_stack.push(clip_index);
    }

    pub fn clear_clip(&mut self) {
        self.clip_stack.pop().unwrap();
    }

    fn add_primitive(&mut self,
                     rect: &Rect<f32>,
                     details: PrimitiveDetails) {
        let current_layer = *self.layer_stack.last().unwrap();
        let StackingContextIndex(layer_index) = current_layer;
        let layer = &mut self.layers[layer_index as usize];

        let prim = Primitive {
            rect: *rect,
            xf_rect: None,
            clip_index: self.clip_stack.last().map(|i| *i),
            details: details,
        };
        layer.primitives.push(prim);
    }

    pub fn push_layer(&mut self,
                      rect: Rect<f32>,
                      _clip_rect: Rect<f32>,
                      transform: Matrix4D<f32>,
                      opacity: f32,
                      pipeline_id: PipelineId,
                      scroll_layer_id: ScrollLayerId,
                      offset: Point2D<f32>) {
        let sc = StackingContext {
            primitives: Vec::new(),
            local_rect: rect,
            local_transform: transform,
            local_offset: offset,
            scroll_layer_id: scroll_layer_id,
            opacity: opacity,
            pipeline_id: pipeline_id,
            xf_rect: None,
            transform: Matrix4D::identity(),
        };

        self.layer_stack.push(StackingContextIndex(self.layers.len()));
        self.layers.push(sc);
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

        let prim = TextPrimitive {
            color: *color,
            font_key: font_key,
            size: size,
            blur_radius: blur_radius,
            glyph_range: glyph_range,
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

        let bs_rect = compute_box_shadow_rect(box_bounds,
                                              box_offset,
                                              spread_radius);

        let metrics = BoxShadowMetrics::new(&bs_rect,
                                            border_radius,
                                            blur_radius);

        let prim_rect = Rect::new(metrics.tl_outer,
                                  Size2D::new(metrics.br_outer.x - metrics.tl_outer.x,
                                              metrics.br_outer.y - metrics.tl_outer.y));

        let prim = BoxShadowPrimitive {
            metrics: metrics,
            src_rect: *box_bounds,
            bs_rect: bs_rect,
            color: *color,
            blur_radius: blur_radius,
            spread_radius: spread_radius,
            border_radius: border_radius,
            clip_mode: clip_mode,
        };

        self.add_primitive(&prim_rect, PrimitiveDetails::BoxShadow(prim));
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

    fn cull_layers(&mut self,
                   screen_rect: &Rect<DevicePixel>,
                   layer_map: &HashMap<ScrollLayerId, Layer, BuildHasherDefault<FnvHasher>>) {
        // Remove layers that are transparent.
        self.layers.retain(|layer| {
            layer.opacity > 0.0
        });

        // Build layer screen rects.
        // TODO(gw): This can be done earlier once update_layer_transforms() is fixed.
        for layer in &mut self.layers {
            let scroll_layer = &layer_map[&layer.scroll_layer_id];
            let offset_transform = Matrix4D::identity().translate(layer.local_offset.x,
                                                                  layer.local_offset.y,
                                                                  0.0);
            let transform = scroll_layer.world_transform
                                        .as_ref()
                                        .unwrap()
                                        .mul(&layer.local_transform)
                                        .mul(&offset_transform);
            layer.transform = transform;
            layer.xf_rect = Some(TransformedRect::new(&layer.local_rect,
                                                      &transform,
                                                      self.device_pixel_ratio));
        }

        self.layers.retain(|layer| {
            layer.xf_rect
                 .as_ref()
                 .unwrap()
                 .bounding_rect
                 .intersects(&screen_rect)
        });

        for layer in &mut self.layers {
            for prim in &mut layer.primitives {
                prim.xf_rect = Some(TransformedRect::new(&prim.rect,
                                                         &layer.transform,
                                                         self.device_pixel_ratio));
            }

            layer.primitives.retain(|prim| {
                prim.xf_rect
                    .as_ref()
                    .unwrap()
                    .bounding_rect
                    .intersects(&screen_rect)
            });
        }
    }

    fn create_screen_tiles(&self) -> Vec<ScreenTile> {
        let dp_size = Size2D::new(DevicePixel::new(self.screen_rect.size.width as f32,
                                                   self.device_pixel_ratio),
                                  DevicePixel::new(self.screen_rect.size.height as f32,
                                                   self.device_pixel_ratio));

        let x_tile_size = DevicePixel(SCREEN_TILE_SIZE as i32);
        let y_tile_size = DevicePixel(SCREEN_TILE_SIZE as i32);
        let x_tile_count = (dp_size.width + x_tile_size - DevicePixel(1)).0 / x_tile_size.0;
        let y_tile_count = (dp_size.height + y_tile_size - DevicePixel(1)).0 / y_tile_size.0;

        // Build screen space tiles, which are individual BSP trees.
        let mut screen_tiles = Vec::new();
        for y in 0..y_tile_count {
            let y0 = DevicePixel(y * y_tile_size.0);
            let y1 = y0 + y_tile_size;

            for x in 0..x_tile_count {
                let x0 = DevicePixel(x * x_tile_size.0);
                let x1 = x0 + x_tile_size;

                let tile_rect = rect_from_points(x0, y0, x1, y1);

                screen_tiles.push(ScreenTile::new(tile_rect));
            }
        }

        screen_tiles
    }

    fn assign_prims_to_screen_tiles(&self,
                                    screen_tiles: &mut Vec<ScreenTile>,
                                    debug_rects: &mut Vec<DebugRect>) { //-> usize {
        //let mut pass_count = 0;

        // TODO(gw): This can be made much faster - calculate tile indices and
        //           assign in a loop.
        for screen_tile in screen_tiles {
            let mut prim_count = 0;
            for (layer_index, layer) in self.layers
                                            .iter()
                                            .enumerate() {
                let layer_index = StackingContextIndex(layer_index);
                let layer_rect = layer.xf_rect.as_ref().unwrap().bounding_rect;

                if layer_rect.intersects(&screen_tile.rect) {
                    let mut tile_layer = ScreenTileLayer {
                        layer_index: layer_index,
                        prim_indices: Vec::new(),
                        layer_in_ubo: 0,
                        layer_opacity: layer.opacity,
                        is_opaque: false,
                    };
                    for (prim_index, prim) in layer.primitives.iter().enumerate() {
                        let prim_rect = &prim.xf_rect.as_ref().unwrap().bounding_rect;
                        if prim_rect.intersects(&screen_tile.rect) {
                            prim_count += 1;
                            tile_layer.prim_indices.push(PrimitiveIndex(prim_index));
                        }
                    }
                    if tile_layer.prim_indices.len() > 0 {
                        tile_layer.compile(layer, &screen_tile.rect);
                        if tile_layer.is_opaque {
                            //println!(" rem {} layers!", screen_tile.layers.len());
                            screen_tile.layers.clear();
                        }
                        screen_tile.layers.push(tile_layer);
                    }
                }
            }

            let is_debug = false;//screen_tile.rect.origin.x.0 == 1024 &&
                           //screen_tile.rect.origin.y.0 == 640;

            if is_debug {
                println!("debug: {:?}", screen_tile.layers);
            }

            if self.debug {
                debug_rects.push(DebugRect {
                    label: format!("{}|{}", screen_tile.layer_count(), prim_count),
                    color: if is_debug {
                        ColorF::new(0.0, 1.0, 0.0, 1.0)
                    } else {
                        ColorF::new(1.0, 0.0, 0.0, 1.0)
                    },
                    rect: screen_tile.rect,
                })
            }
        }

        //pass_count
    }

    fn build_resource_list(&mut self,
                           resource_cache: &mut ResourceCache,
                           frame_id: FrameId,
                           pipeline_auxiliary_lists: &HashMap<PipelineId, AuxiliaryLists, BuildHasherDefault<FnvHasher>>) {
        let mut resource_list = ResourceList::new(self.device_pixel_ratio);

        // Non-visible layers have been removed by now
        for layer in &self.layers {
            let auxiliary_lists = pipeline_auxiliary_lists.get(&layer.pipeline_id)
                                                          .expect("No auxiliary lists?!");

            // Non-visible chunks have also been removed by now
            layer.build_resource_list(&mut resource_list,
                                      //&layer.primitives,
                                      auxiliary_lists);
        }

        resource_cache.add_resource_list(&resource_list,
                                         frame_id);
        resource_cache.raster_pending_glyphs(frame_id);
    }

    pub fn build(&mut self,
                 resource_cache: &mut ResourceCache,
                 frame_id: FrameId,
                 pipeline_auxiliary_lists: &HashMap<PipelineId, AuxiliaryLists, BuildHasherDefault<FnvHasher>>,
                 layer_map: &HashMap<ScrollLayerId, Layer, BuildHasherDefault<FnvHasher>>) -> Frame {
        let screen_rect = Rect::new(Point2D::zero(),
                                    Size2D::new(DevicePixel::new(self.screen_rect.size.width as f32, self.device_pixel_ratio),
                                                DevicePixel::new(self.screen_rect.size.height as f32, self.device_pixel_ratio)));

        self.cull_layers(&screen_rect, layer_map);

        let mut debug_rects = Vec::new();
        let mut screen_tiles = self.create_screen_tiles();

        self.assign_prims_to_screen_tiles(&mut screen_tiles,
                                          &mut debug_rects);

        self.build_resource_list(resource_cache,
                                 frame_id,
                                 pipeline_auxiliary_lists);

        let mut clear_tiles = Vec::new();
        let mut error_tiles = Vec::new();

        let mut current_phase = RenderPhase::new(self.layers.len());

        // Build list of passes, target allocs that each tile needs.
        for screen_tile in &mut screen_tiles {
            screen_tile.build_resource_list();
        }

        // Sort by pass count to minimize render target switches.
        screen_tiles.sort_by(|a, b| {
            let a_passes = a.resource_list.as_ref().map_or(0, |a| a.passes.len());
            let b_passes = b.resource_list.as_ref().map_or(0, |b| b.passes.len());
            b_passes.cmp(&a_passes)
        });

        for screen_tile in &mut screen_tiles {
            println!("\t{:?}", screen_tile.resource_list.as_ref().map_or(0, |a| a.passes.len()));

            let kind = screen_tile.compile(&self.layers,
                                           pipeline_auxiliary_lists,
                                           resource_cache,
                                           frame_id,
                                           self.device_pixel_ratio,
                                           &mut current_phase);

            match kind {
                ScreenTileCompileResult::Clear => {
                    clear_tiles.push(ClearTile {
                        rect: screen_tile.rect,
                    });
                }
                ScreenTileCompileResult::Unhandled => {
                    error_tiles.push(ErrorTile {
                        rect: screen_tile.rect,
                    });
                }
                ScreenTileCompileResult::OutOfMemory => {
                    panic!("todo");
                }
                ScreenTileCompileResult::Ok => {
                }
            }
        }

        current_phase.build();

        Frame {
            debug_rects: debug_rects,
            phases: vec![current_phase],
            clear_tiles: clear_tiles,
            error_tiles: error_tiles,
            cache_size: Size2D::new(RENDERABLE_CACHE_SIZE.0 as f32,
                                    RENDERABLE_CACHE_SIZE.0 as f32),
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
