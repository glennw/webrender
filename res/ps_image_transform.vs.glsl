#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

struct Image {
    PrimitiveInfo info;
    vec4 local_rect;
    vec4 st_rect;
};

layout(std140) uniform Items {
    Image images[1024];
};

void main(void) {
    Image image = images[gl_InstanceID];
    Layer layer = layers[image.info.layer_tile_part.x];
    Tile tile = tiles[image.info.layer_tile_part.y];

    vec2 p0 = image.local_rect.xy;
    vec2 p1 = image.local_rect.xy + vec2(image.local_rect.z, 0.0);
    vec2 p2 = image.local_rect.xy + vec2(0.0, image.local_rect.w);
    vec2 p3 = image.local_rect.xy + image.local_rect.zw;

    vec4 t0 = layer.transform * vec4(p0, 0, 1);
    vec4 t1 = layer.transform * vec4(p1, 0, 1);
    vec4 t2 = layer.transform * vec4(p2, 0, 1);
    vec4 t3 = layer.transform * vec4(p3, 0, 1);

    vec2 tp0 = t0.xy / t0.w;
    vec2 tp1 = t1.xy / t1.w;
    vec2 tp2 = t2.xy / t2.w;
    vec2 tp3 = t3.xy / t3.w;

    vec2 min_pos = min(tp0.xy, min(tp1.xy, min(tp2.xy, tp3.xy)));
    vec2 max_pos = max(tp0.xy, max(tp1.xy, max(tp2.xy, tp3.xy)));

    vec2 min_pos_clamped = clamp(min_pos * uDevicePixelRatio,
                                 tile.actual_rect.xy,
                                 tile.actual_rect.xy + tile.actual_rect.zw);

    vec2 max_pos_clamped = clamp(max_pos * uDevicePixelRatio,
                                 tile.actual_rect.xy,
                                 tile.actual_rect.xy + tile.actual_rect.zw);

    vec2 clamped_pos = mix(min_pos_clamped,
                           max_pos_clamped,
                           aPosition.xy);

    vec3 layer_pos = get_layer_pos(clamped_pos / uDevicePixelRatio, image.info.layer_tile_part.x);

    vRect = image.local_rect;
    vPos = layer_pos;

    vec2 f = (layer_pos.xy - image.local_rect.xy) / image.local_rect.zw;

    vUv = mix(image.st_rect.xy,
              image.st_rect.zw,
              f);

    vec2 final_pos = clamped_pos + tile.target_rect.xy - tile.actual_rect.xy;

    gl_Position = uTransform * vec4(final_pos, 0, 1);
}
