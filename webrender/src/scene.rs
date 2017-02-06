/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use fnv::FnvHasher;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use tiling::AuxiliaryListsMap;
use webrender_traits::{AuxiliaryLists, BuiltDisplayList, PipelineId, Epoch, ColorF};
use webrender_traits::{DisplayItem, LayerSize, LayoutTransform};
use webrender_traits::{FloatProperty, LayoutTransformProperty, PropertyBindingKey, PropertyValue, SpecificPropertyValue};

/// Stores a map of the animated property bindings for the current display list. These
/// can be used to animate the transform and/or opacity of a display list without
/// re-submitting the display list itself.
pub struct SceneProperties {
    properties: HashMap<PropertyBindingKey, SpecificPropertyValue, BuildHasherDefault<FnvHasher>>,
}

impl SceneProperties {
    pub fn new() -> SceneProperties {
        SceneProperties {
            properties: HashMap::with_hasher(Default::default()),
        }
    }

    /// Set the current property list for this display list.
    pub fn set_properties(&mut self, values: Vec<PropertyValue>) {
        self.properties.clear();

        for property in values {
            self.properties.insert(property.key, property.value);
        }
    }

    /// Get the current value for a transform property.
    pub fn resolve_layout_transform(&self, property: &LayoutTransformProperty) -> LayoutTransform {
        match *property {
            LayoutTransformProperty::Value(matrix) => matrix,
            LayoutTransformProperty::Binding(ref key) => {
                match self.properties.get(key) {
                    Some(&SpecificPropertyValue::LayoutTransform(matrix)) => matrix,
                    Some(..) | None => {
                        warn!("Property binding {:?} has an invalid value.", key);
                        LayoutTransform::identity()
                    }
                }
            }
        }
    }

    /// Get the current value for a float property.
    pub fn resolve_float(&self, property: &FloatProperty, default_value: f32) -> f32 {
        match *property {
            FloatProperty::Value(value) => value,
            FloatProperty::Binding(ref key) => {
                match self.properties.get(key) {
                    Some(&SpecificPropertyValue::Float(value)) => value,
                    Some(..) | None => {
                        warn!("Property binding {:?} has an invalid value.", key);
                        default_value
                    }
                }
            }
        }
    }
}

/// A representation of the layout within the display port for a given document or iframe.
#[derive(Debug)]
pub struct ScenePipeline {
    pub pipeline_id: PipelineId,
    pub epoch: Epoch,
    pub viewport_size: LayerSize,
    pub background_color: Option<ColorF>,
}

/// A complete representation of the layout bundling visible pipelines together.
pub struct Scene {
    pub root_pipeline_id: Option<PipelineId>,
    pub pipeline_map: HashMap<PipelineId, ScenePipeline, BuildHasherDefault<FnvHasher>>,
    pub pipeline_auxiliary_lists: AuxiliaryListsMap,
    pub display_lists: HashMap<PipelineId, Vec<DisplayItem>, BuildHasherDefault<FnvHasher>>,
    pub properties: SceneProperties,
}

impl Scene {
    pub fn new() -> Scene {
        Scene {
            root_pipeline_id: None,
            pipeline_map: HashMap::with_hasher(Default::default()),
            pipeline_auxiliary_lists: HashMap::with_hasher(Default::default()),
            display_lists: HashMap::with_hasher(Default::default()),
            properties: SceneProperties::new(),
        }
    }

    pub fn set_root_pipeline_id(&mut self, pipeline_id: PipelineId) {
        self.root_pipeline_id = Some(pipeline_id);
    }

    pub fn set_root_display_list(&mut self,
                                 pipeline_id: PipelineId,
                                 epoch: Epoch,
                                 built_display_list: BuiltDisplayList,
                                 background_color: Option<ColorF>,
                                 viewport_size: LayerSize,
                                 auxiliary_lists: AuxiliaryLists) {
        self.pipeline_auxiliary_lists.insert(pipeline_id, auxiliary_lists);
        self.display_lists.insert(pipeline_id, built_display_list.all_display_items().to_vec());

        let new_pipeline = ScenePipeline {
            pipeline_id: pipeline_id,
            epoch: epoch,
            viewport_size: viewport_size,
            background_color: background_color,
        };

        self.pipeline_map.insert(pipeline_id, new_pipeline);
    }
}
