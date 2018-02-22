/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#define VECS_PER_SPECIFIC_BRUSH 8

#include shared,prim_shared,brush

flat varying vec4 vColor;

#ifdef WR_FEATURE_ALPHA_PASS
varying vec2 vLocalPos;
#endif

#ifdef WR_VERTEX_SHADER

void brush_vs(
    VertexInfo vi,
    int prim_address,
    RectWithSize local_rect,
    ivec3 user_data,
    PictureTask pic_task
) {
#ifdef WR_FEATURE_ALPHA_PASS
    vLocalPos = vi.local_pos;
#endif
}
#endif

#ifdef WR_FRAGMENT_SHADER
vec4 brush_fs() {
    vec4 color = vec4(1,0,0,1);

#ifdef WR_FEATURE_ALPHA_PASS
    color *= init_transform_fs(vLocalPos);
#endif

    return color;
}
#endif
