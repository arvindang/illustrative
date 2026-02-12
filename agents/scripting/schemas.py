"""
JSON response schemas for the ScriptingAgent passes.

Each schema defines the structure expected from Gemini API responses
using the google.genai structured output format.
"""

# =============================================================================
# PASS 1: Beat Analysis Schema
# =============================================================================
BEAT_ANALYSIS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "total_word_count": {"type": "INTEGER"},
        "beats": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "beat_id": {"type": "INTEGER"},
                    "beat_type": {"type": "STRING"},
                    "description": {"type": "STRING"},
                    "intensity": {"type": "NUMBER"},
                    "key_characters": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "emotional_tone": {"type": "STRING"},
                    "scene_type": {"type": "STRING"},
                    # Visual adaptation fields
                    "visual_potential": {"type": "NUMBER"},
                    "adaptation_notes": {"type": "STRING"},
                    "suggested_focus": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "micro_beats": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "is_page_turn_hook": {"type": "BOOLEAN"}
                },
                "required": ["beat_id", "beat_type", "description", "intensity", "key_characters", "visual_potential"]
            }
        },
        "act_boundaries": {
            "type": "OBJECT",
            "properties": {
                "act_1_end": {"type": "INTEGER"},
                "act_2_end": {"type": "INTEGER"}
            }
        },
        "hooks": {"type": "ARRAY", "items": {"type": "INTEGER"}},
        "low_visual_warnings": {"type": "ARRAY", "items": {"type": "INTEGER"}}
    },
    "required": ["beats", "act_boundaries"]
}

# =============================================================================
# PASS 1.5: Adaptation Filter Schema
# =============================================================================
ADAPTATION_FILTER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "essential_scenes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "description": {"type": "STRING"},
                    "beat_ids": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    "why_essential": {"type": "STRING"}
                },
                "required": ["description", "beat_ids", "why_essential"]
            }
        },
        "condensable_scenes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "description": {"type": "STRING"},
                    "beat_ids": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    "condensation_strategy": {"type": "STRING"}
                },
                "required": ["description", "beat_ids", "condensation_strategy"]
            }
        },
        "cuttable_scenes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "description": {"type": "STRING"},
                    "beat_ids": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    "reason_to_cut": {"type": "STRING"}
                },
                "required": ["description", "beat_ids", "reason_to_cut"]
            }
        },
        "creative_adaptations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "description": {"type": "STRING"},
                    "beat_ids": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    "adaptation_strategy": {"type": "STRING"}
                },
                "required": ["description", "beat_ids", "adaptation_strategy"]
            }
        },
        "reader_beloved_moments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "description": {"type": "STRING"},
                    "beat_ids": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    "why_beloved": {"type": "STRING"}
                },
                "required": ["description", "why_beloved"]
            }
        },
        "pacing_recommendations": {
            "type": "OBJECT",
            "properties": {
                "slow_down": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                "speed_up": {"type": "ARRAY", "items": {"type": "INTEGER"}}
            }
        },
        "natural_chapter_breaks": {"type": "ARRAY", "items": {"type": "INTEGER"}}
    },
    "required": ["essential_scenes", "condensable_scenes", "cuttable_scenes", "reader_beloved_moments"]
}

# =============================================================================
# PASS 2: Pacing Blueprint Schema
# =============================================================================
BLUEPRINT_PAGE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "page_number": {"type": "INTEGER"},
            "summary": {"type": "STRING"},
            "focus_text": {"type": "STRING"},
            "mood": {"type": "STRING"},
            "key_characters": {"type": "ARRAY", "items": {"type": "STRING"}},
            "visual_notes": {"type": "STRING"},
            "scene_type": {"type": "STRING"},
            # Full-bleed and pacing fields (for digital reading - one page at a time)
            "is_spread": {"type": "BOOLEAN"},  # Full-bleed single page (epic moments)
            "is_cliffhanger": {"type": "BOOLEAN"},  # Page ends with hook/reveal
            "page_turn_note": {"type": "STRING"},
            "suggested_panel_count": {"type": "INTEGER"},
            "recommended_splash": {"type": "BOOLEAN"}
        },
        "required": ["page_number", "summary", "focus_text", "key_characters", "scene_type"]
    }
}

# =============================================================================
# PASS 3: Character Deep Dive Schema
# =============================================================================
CHARACTER_DEEP_DIVE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "characters": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "role": {"type": "STRING"},
                    "arc_type": {"type": "STRING"},
                    "introduction_page": {"type": "INTEGER"},
                    "distinctive_items": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "era_appropriate_gear": {
                        "type": "OBJECT",
                        "additionalProperties": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"}
                        }
                    },
                    "relationships": {
                        "type": "OBJECT",
                        "additionalProperties": {"type": "STRING"}
                    },
                    "key_moments": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "page": {"type": "INTEGER"},
                                "event": {"type": "STRING"},
                                "emotional_state": {"type": "STRING"},
                                "visual_change": {"type": "STRING"}
                            }
                        }
                    },
                    # Voice/dialect tracking fields
                    "voice_profile": {
                        "type": "OBJECT",
                        "properties": {
                            "education_level": {"type": "STRING"},
                            "formality": {"type": "STRING"},
                            "vocabulary_style": {"type": "STRING"},
                            "dialect_markers": {"type": "ARRAY", "items": {"type": "STRING"}},
                            "emotional_tells": {"type": "STRING"},
                            "catchphrases": {"type": "ARRAY", "items": {"type": "STRING"}}
                        }
                    },
                    "dialogue_samples": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "speech_contrast": {"type": "STRING"}
                },
                "required": ["name", "role", "distinctive_items"]
            }
        },
        "scene_states": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "page_number": {"type": "INTEGER"},
                    "characters": {
                        "type": "OBJECT",
                        "additionalProperties": {
                            "type": "OBJECT",
                            "properties": {
                                "emotional_state": {"type": "STRING"},
                                "gear": {"type": "ARRAY", "items": {"type": "STRING"}},
                                "notes": {"type": "STRING"}
                            }
                        }
                    },
                    "interaction_rules": {"type": "ARRAY", "items": {"type": "STRING"}}
                },
                "required": ["page_number", "characters"]
            }
        }
    },
    "required": ["characters", "scene_states"]
}

# =============================================================================
# PASS 4: Asset Manifest Schema
# =============================================================================
ASSET_MANIFEST_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "characters": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "aliases": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "description": {"type": "STRING"},
                    "age_range": {"type": "STRING"},
                    "occupation": {"type": "STRING"},
                    "distinctive_items": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "specific_era_markers": {"type": "STRING"},
                    "color_signature": {"type": "ARRAY", "items": {"type": "STRING"}}
                },
                "required": ["name", "description", "aliases"]
            }
        },
        "objects": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "description": {"type": "STRING"},
                    "key_features": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "condition": {"type": "STRING"},
                    "material_context": {"type": "STRING"}
                },
                "required": ["name", "description"]
            }
        },
        "locations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "description": {"type": "STRING"},
                    "color_palette": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "lighting": {"type": "STRING"},
                    "recurring_elements": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "mood": {"type": "STRING"},
                    "era_markers": {"type": "STRING"}
                },
                "required": ["name", "description", "lighting", "mood"]
            }
        },
        "interaction_rules": {
            "type": "OBJECT",
            "properties": {
                "underwater_scenes": {"type": "ARRAY", "items": {"type": "STRING"}},
                "formal_scenes": {"type": "ARRAY", "items": {"type": "STRING"}},
                "action_scenes": {"type": "ARRAY", "items": {"type": "STRING"}},
                "aboard_ship": {"type": "ARRAY", "items": {"type": "STRING"}}
            }
        },
        "forbidden_combinations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "characters": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "items": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "reason": {"type": "STRING"}
                }
            }
        },
        "color_script": {
            "type": "OBJECT",
            "properties": {
                "primary_palette": {"type": "ARRAY", "items": {"type": "STRING"}},
                "act_1_colors": {"type": "ARRAY", "items": {"type": "STRING"}},
                "act_2_colors": {"type": "ARRAY", "items": {"type": "STRING"}},
                "act_3_colors": {"type": "ARRAY", "items": {"type": "STRING"}},
                "color_associations": {
                    "type": "OBJECT",
                    "additionalProperties": {"type": "STRING"}
                }
            }
        }
    },
    "required": ["characters", "objects", "locations", "interaction_rules"]
}

# =============================================================================
# PASS 5: Page Script Schema
# =============================================================================
PAGE_SCRIPT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "page_number": {"type": "INTEGER"},
        # Layout metadata (preserved from blueprint for downstream agents)
        "recommended_splash": {"type": "BOOLEAN"},
        "is_full_bleed": {"type": "BOOLEAN"},  # Full-bleed single page (for digital reading)
        "is_cliffhanger": {"type": "BOOLEAN"},
        "suggested_panel_count": {"type": "INTEGER"},
        "scene_type": {"type": "STRING"},
        "panels": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "panel_id": {"type": "INTEGER"},
                    "visual_description": {"type": "STRING"},
                    "dialogue": {"type": "STRING"},
                    "dialogue_bubbles": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "speaker": {"type": "STRING"},
                                "text": {"type": "STRING"},
                                "position": {
                                    "type": "STRING",
                                    "enum": ["top-left", "top-right", "bottom-left", "bottom-right"]
                                }
                            },
                            "required": ["speaker", "text", "position"]
                        }
                    },
                    "caption": {"type": "STRING"},
                    "characters": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "key_objects": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "bubble_position": {
                        "type": "STRING",
                        "enum": ["top-left", "top-right", "bottom-left", "bottom-right", "caption-box"]
                    },
                    # Cinematic fields
                    "shot_type": {
                        "type": "STRING",
                        "enum": ["establishing", "wide", "medium", "close-up", "extreme-close-up",
                                 "over-shoulder", "two-shot", "birds-eye", "worms-eye"]
                    },
                    "panel_size": {
                        "type": "STRING",
                        "enum": ["large", "medium", "small"]
                    },
                    "advice": {
                        "type": "OBJECT",
                        "properties": {
                            "scene_type": {"type": "STRING"},
                            "required_gear": {
                                "type": "OBJECT",
                                "additionalProperties": {
                                    "type": "ARRAY",
                                    "items": {"type": "STRING"}
                                }
                            },
                            "era_constraints": {"type": "ARRAY", "items": {"type": "STRING"}},
                            "continuity": {
                                "type": "OBJECT",
                                "properties": {
                                    "from_previous": {"type": "STRING"},
                                    "to_next": {"type": "STRING"}
                                }
                            },
                            "composition": {
                                "type": "OBJECT",
                                "properties": {
                                    "negative_space": {"type": "STRING"}
                                }
                            }
                        }
                    }
                },
                "required": ["panel_id", "visual_description", "characters", "shot_type", "panel_size", "advice"]
            }
        }
    },
    "required": ["page_number", "panels"]
}

# =============================================================================
# PASS 5.5: Dialogue Polish Schema
# =============================================================================
DIALOGUE_POLISH_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "page_number": {"type": "INTEGER"},
            "panels": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "panel_id": {"type": "INTEGER"},
                        "dialogue": {"type": "STRING"},
                        "dialogue_bubbles": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "speaker": {"type": "STRING"},
                                    "text": {"type": "STRING"},
                                    "position": {"type": "STRING"}
                                }
                            }
                        },
                        "caption": {"type": "STRING"},
                        "changes_made": {"type": "STRING"}
                    },
                    "required": ["panel_id"]
                }
            }
        },
        "required": ["page_number", "panels"]
    }
}
