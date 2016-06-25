/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use bsptree::BspTree;
use euclid::Rect;
use internal_types::DevicePixel;
use std::cmp;
use tiling::{BuiltTile, RenderableId, RenderableInstanceId, RenderableList, TileRange};
use tiling::{TilingStrategy};
use util;

#[derive(Debug)]
struct TilingInfo {
    x_tile_count: i32,
    y_tile_count: i32,
    x_tile_size: DevicePixel,
    y_tile_size: DevicePixel,
}

impl TilingInfo {
    #[inline(always)]
    fn get_tile_range(&self, rect: &Rect<DevicePixel>) -> TileRange {
        let px0 = rect.origin.x;
        let py0 = rect.origin.y;
        let px1 = rect.origin.x + rect.size.width;
        let py1 = rect.origin.y + rect.size.height;

        let tx0 = px0.0 / self.x_tile_size.0;
        let ty0 = py0.0 / self.y_tile_size.0;
        let tx1 = (px1.0 + self.x_tile_size.0 - 1) / self.x_tile_size.0;
        let ty1 = (py1.0 + self.y_tile_size.0 - 1) / self.y_tile_size.0;

        let tx0 = cmp::max(0, cmp::min(tx0, self.x_tile_count));
        let ty0 = cmp::max(0, cmp::min(ty0, self.y_tile_count));
        let tx1 = cmp::max(0, cmp::min(tx1, self.x_tile_count));
        let ty1 = cmp::max(0, cmp::min(ty1, self.y_tile_count));

        TileRange::new(tx0, ty0, tx1, ty1)
    }
}

struct Tile {
    bsp_tree: BspTree<RenderableInstanceId>,
    rect: Rect<DevicePixel>,
    instances: Vec<RenderableId>,
}

impl Tile {
    fn new(x0: DevicePixel,
           y0: DevicePixel,
           x1: DevicePixel,
           y1: DevicePixel)
           -> Tile {
        let tile_rect = util::rect_from_points(x0, y0, x1, y1);
        Tile {
            rect: tile_rect,
            bsp_tree: BspTree::new(tile_rect),
            instances: Vec::new(),
        }
    }

    fn add(&mut self, bounding_rect: &Rect<DevicePixel>, id: RenderableId) {
        let instance_id = RenderableInstanceId(self.instances.len() as u32);
        self.instances.push(id);
        self.bsp_tree.add(bounding_rect, instance_id);
    }
}

pub struct BspTilingStrategy {
    tiling_info: TilingInfo,
    tiles: Vec<Tile>,
    device_pixel_ratio: f32,
}

impl BspTilingStrategy {
    pub fn new(screen_rect: &Rect<DevicePixel>, device_pixel_ratio: f32) -> BspTilingStrategy {
        let x_tile_size = DevicePixel(512);
        let y_tile_size = DevicePixel(512);
        let x_tile_count = (screen_rect.size.width + x_tile_size - DevicePixel(1)).0 /
            x_tile_size.0;
        let y_tile_count = (screen_rect.size.height + y_tile_size - DevicePixel(1)).0 /
            y_tile_size.0;

        let tile_info = TilingInfo {
           x_tile_count: x_tile_count,
           y_tile_count: y_tile_count,
           x_tile_size: x_tile_size,
           y_tile_size: y_tile_size,
        };

        BspTilingStrategy {
            tiling_info: tile_info,
            tiles: vec![],
            device_pixel_ratio: device_pixel_ratio,
        }
    }
}

impl TilingStrategy for BspTilingStrategy {
    fn add_renderables(&mut self, rlist: &RenderableList) {
        // Build screen space tiles, which are individual BSP trees.
        for y in 0..self.tiling_info.y_tile_count {
            let y0 = DevicePixel(y * self.tiling_info.y_tile_size.0);
            let y1 = y0 + self.tiling_info.y_tile_size;

            for x in 0..self.tiling_info.x_tile_count {
                let x0 = DevicePixel(x * self.tiling_info.x_tile_size.0);
                let x1 = x0 + self.tiling_info.x_tile_size;

                self.tiles.push(Tile::new(x0, y0, x1, y1));
            }
        }

        // Add each of the visible primitives to each BSP tile that it touches.
        for (renderable_index, renderable) in rlist.renderables.iter().enumerate() {
            let renderable_id = RenderableId(renderable_index as u32);
            let tile_range = self.tiling_info.get_tile_range(&renderable.bounding_rect);

            for y in tile_range.y0..tile_range.y1 {
                for x in tile_range.x0..tile_range.x1 {
                    let tile = &mut self.tiles[(y*self.tiling_info.x_tile_count+x) as usize];
                    tile.add(&renderable.bounding_rect, renderable_id);
                }
            }
        }
    }

    fn region_count(&mut self) -> usize {
        self.tiles.len()
    }

    #[inline(always)]
    fn get_tile_range(&self, rect: &Rect<DevicePixel>) -> TileRange {
        self.tiling_info.get_tile_range(rect)
    }

    fn build_and_process_tiles<F>(&mut self, region_index: usize, mut iteration_function: F)
                                  where F: for<'a> FnMut(Rect<DevicePixel>,
                                                         BuiltTile<'a>,
                                                         &'a mut Vec<RenderableId>) {
        let tile = &mut self.tiles[region_index];
        let bsp_tree = &mut tile.bsp_tree;
        let instances = &mut tile.instances;
        bsp_tree.split(self.device_pixel_ratio, &mut |rect, mut cover_indices, partial_indices| {
            if partial_indices.is_empty() {
                cover_indices.sort_by(|a, b| b.cmp(&a));
                iteration_function(*rect, BuiltTile::Tile(cover_indices), instances)
            } else {
                iteration_function(*rect, BuiltTile::Error, instances)
            }
        });
    }

    fn instances(&mut self, region_index: usize) -> &mut Vec<RenderableId> {
        &mut self.tiles[region_index].instances
    }
}

