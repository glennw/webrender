/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared

varying vec2 vUv;

#ifdef WR_VERTEX_SHADER

in int aTaskAddress;
in vec4 aLocalRect;

void main(void) {
    RenderTaskCommonData task = fetch_render_task_common_data(aTaskAddress);

    // todo: !!!!!!!!!!!!!!! subpx sizes, device pixel ratio, scale factors!!
    vec2 pos = task.task_rect.p0 +
               aLocalRect.xy +
               aLocalRect.zw * aPosition.xy;

    vUv = aPosition.xy;

    gl_Position = uTransform * vec4(pos, 0.0, 1.0);
}
#endif

#ifdef WR_FRAGMENT_SHADER
void main(void) {
    oFragColor = vec4(vUv,0,1);
}
#endif
