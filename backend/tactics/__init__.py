"""
HALO Tactics module.

Turns the raw event/commentary/KPI output already produced by
pipeline.VideoProcessor into a coach-style, both-teams-equal tactical
analysis using the Gemini API (default model: gemini-2.5-flash).

This module never trains or fine-tunes anything -- it is a thin,
well-prompted layer on top of the real detection/tracking data your
pipeline already computes, exactly as recommended in the ChatGPT plan
this project follows. It intentionally produces MATCH ANALYSIS ONLY
(no training drills / coaching session plans).
"""
