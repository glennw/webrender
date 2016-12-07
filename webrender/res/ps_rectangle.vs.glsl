#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

VertexInfo write_vertex2(vec4 instance_rect,
                         vec2 offset,
                         vec4 local_clip_rect,
                         float z,
                         Layer layer,
                         Tile tile) {
    vec2 p0 = floor(0.5 + instance_rect.xy * uDevicePixelRatio) / uDevicePixelRatio;
    vec2 p1 = floor(0.5 + (instance_rect.xy + instance_rect.zw) * uDevicePixelRatio) / uDevicePixelRatio;

    vec2 local_pos = mix(p0, p1, aPosition.xy) + offset;

    vec2 cp0 = floor(0.5 + local_clip_rect.xy * uDevicePixelRatio) / uDevicePixelRatio;
    vec2 cp1 = floor(0.5 + (local_clip_rect.xy + local_clip_rect.zw) * uDevicePixelRatio) / uDevicePixelRatio;
    local_pos = clamp(local_pos, cp0, cp1);

    local_pos = clamp_rect(local_pos, layer.local_clip_rect);

    vec4 world_pos = layer.transform * vec4(local_pos, 0.0, 1.0);
    world_pos.xyz /= world_pos.w;

    vec2 device_pos = world_pos.xy * uDevicePixelRatio;

    vec2 clamped_pos = clamp(device_pos,
                             tile.screen_origin_task_origin.xy,
                             tile.screen_origin_task_origin.xy + tile.size_target_index.xy);
    clamped_pos = device_pos;

    vec4 local_clamped_pos = layer.inv_transform * vec4(clamped_pos / uDevicePixelRatio, world_pos.z, 1);
    local_clamped_pos.xyz /= local_clamped_pos.w;

    vec2 final_pos = clamped_pos + tile.screen_origin_task_origin.zw - tile.screen_origin_task_origin.xy;

    gl_Position = uTransform * vec4(final_pos, z, 1);

    VertexInfo vi = VertexInfo(Rect(p0, p1), local_clamped_pos.xy, clamped_pos.xy);
    return vi;
}

void main(void) {
    Primitive prim = load_primitive(gl_InstanceID);
    Rectangle rect = fetch_rectangle(prim.prim_index);

    vColor = rect.color;

#ifdef WR_FEATURE_TRANSFORM
    TransformVertexInfo vi = write_transform_vertex(prim.local_rect,
                                                    prim.local_clip_rect,
                                                    prim.layer,
                                                    prim.tile);
    vLocalRect = vi.clipped_local_rect;
    vLocalPos = vi.local_pos;
#else
    write_vertex2(prim.local_rect,
                                  rect.v_offset[gl_VertexID],
                                 prim.local_clip_rect,
                                 rect.z,
                                 prim.layer,
                                 prim.tile);
#endif

#ifdef WR_FEATURE_CLIP
    write_clip(vi.global_clamped_pos, prim.clip_area);
#endif
}
