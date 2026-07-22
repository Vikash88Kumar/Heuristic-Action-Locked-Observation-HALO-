"""
Unified Data Layer and Tactical Visualization Module
---------------------------------------------------
1. Unified Data Layer: Converts YOLO detections + ByteTrack + Jersey OCR + Team Color into
   one long-format Pandas DataFrame: [frame_id, timestamp, team, player_id, x, y, role, conf]
   with a ball row per frame, normalized once to a fixed 2D pitch coordinate system (0..105m x 0..68m).
2. Static Pitch Validator: Plots 3-4 frames statically using mplsoccer & side-by-side video keyframe.
3. Pygame Tactical Renderer: Strict frame-by-frame draw calls with fixed colors, pass lines, shot rings,
   and commentary text overlays.
4. MoviePy MP4 Exporter: Encodes Pygame frames into an MP4 synced at the original clip's FPS.
5. Plotly HTML Debug Exporter: Interactive HTML dashboard for rapid pipeline iteration.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplsoccer
import pygame
import plotly.graph_objects as go
from pathlib import Path

# Fix Pygame headless display initialization on Windows/Linux environments without GUI
os.environ["SDL_VIDEODRIVER"] = "dummy"

PITCH_LENGTH = 105.0  # meters
PITCH_WIDTH = 68.0    # meters

# Color Palettes
COLOR_MAP = {
    'team_a': (255, 59, 48),    # Bright Red (RGB)
    'team_b': (0, 122, 255),    # Bright Blue (RGB)
    'referee': (255, 204, 0),   # Yellow (RGB)
    'ball': (255, 215, 0),      # Gold (RGB)
    'unknown': (142, 142, 147)  # Grey (RGB)
}

COLOR_MAP_HEX = {
    'team_a': '#FF3B30',
    'team_b': '#007AFF',
    'referee': '#FFCC00',
    'ball': '#FFD700',
    'unknown': '#8E8E93'
}


def build_unified_dataframe(frame_records, jersey_map, final_team, frame_w, frame_h, fps=25.0,
                            pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH, class_roles=None):
    """
    Convert raw frame tracking records into one long-format pandas DataFrame.
    Normalizes x/y once here to a fixed pitch coordinate system (meters).

    Columns: frame_id, timestamp, team, player_id, track_id, x, y, role, conf
    """
    rows = []
    if class_roles is None:
        class_roles = {0: 'player', 1: 'goalkeeper', 2: 'referee', 3: 'ball'}

    for rec in frame_records:
        frame_idx = rec['frame_id']
        ts = round(frame_idx / max(1.0, fps), 3)

        # 1. Process tracked player / referee detections
        for tid, cls_id, (x1, y1, x2, y2) in rec.get('dets', []):
            role_name = class_roles.get(cls_id, 'player')
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Normalize to pitch coordinate system (0..pitch_length, 0..pitch_width)
            pitch_x = round(float((cx / frame_w) * pitch_length), 3)
            pitch_y = round(float((cy / frame_h) * pitch_width), 3)

            if role_name in ('player', 'goalkeeper'):
                t_label = final_team.get(tid, 'team_a')
                if t_label is None or t_label not in ('team_a', 'team_b'):
                    t_label = 'team_a'
                p_id = jersey_map.get(tid, str(tid))
            elif role_name == 'referee':
                t_label = 'referee'
                p_id = f"ref_{tid}"
            else:
                continue

            rows.append({
                'frame_id': int(frame_idx),
                'timestamp': ts,
                'team': t_label,
                'player_id': str(p_id),
                'track_id': int(tid),
                'x': pitch_x,
                'y': pitch_y,
                'role': role_name,
                'conf': round(float(rec.get('conf', 1.0)), 2)
            })

        # 2. Add Ball Row per frame
        roi = rec.get('roi', {})
        ball_center = roi.get('center', (frame_w / 2.0, frame_h / 2.0))
        bx, by = ball_center
        ball_px = round(float((bx / frame_w) * pitch_length), 3)
        ball_py = round(float((by / frame_h) * pitch_width), 3)

        rows.append({
            'frame_id': int(frame_idx),
            'timestamp': ts,
            'team': 'ball',
            'player_id': 'ball',
            'track_id': -1,
            'x': ball_px,
            'y': ball_py,
            'role': 'ball',
            'conf': 1.0
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=['frame_id', 'team', 'player_id']).reset_index(drop=True)
    return df


def validate_pitch_frames(df, source_video_path, output_dir, sample_frames=None,
                          pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH):
    """
    Pick 3-4 frames spread across the clip, plot them statically using mplsoccer,
    and side-by-side with original video frames for visual verification.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_paths = []

    if df.empty:
        print("Warning: Empty DataFrame passed to validate_pitch_frames.")
        return validation_paths

    available_frames = sorted(df['frame_id'].unique())
    if not sample_frames:
        n_total = len(available_frames)
        if n_total <= 4:
            sample_frames = available_frames
        else:
            indices = np.linspace(0, n_total - 1, 4, dtype=int)
            sample_frames = [available_frames[i] for i in indices]

    # Open video capture for source video frame comparison
    cap = cv2.VideoCapture(str(source_video_path))
    video_frames = {}
    current_f = 0
    while cap.isOpened() and len(video_frames) < len(sample_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if current_f in sample_frames:
            video_frames[current_f] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_f += 1
    cap.release()

    pitch = mplsoccer.Pitch(
        pitch_type='custom',
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        pitch_color='#1E3A1E',
        line_color='#FFFFFF',
        linewidth=2
    )

    for fid in sample_frames:
        frame_df = df[df['frame_id'] == fid]
        if frame_df.empty:
            continue

        fig, (ax_vid, ax_pitch) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 1]})

        # Left: Original source video frame
        if fid in video_frames:
            ax_vid.imshow(video_frames[fid])
            ax_vid.set_title(f"Source Video - Frame {fid}", fontsize=14, fontweight='bold')
        else:
            ax_vid.text(0.5, 0.5, f"Frame {fid} Video Unavailable", ha='center', va='center')
        ax_vid.axis('off')

        # Right: mplsoccer 2D pitch view
        pitch.draw(ax=ax_pitch)
        ax_pitch.set_title(f"Normalized Pitch Data - Frame {fid}", fontsize=14, fontweight='bold')

        # Scatter points per team
        for team_name in ['team_a', 'team_b', 'referee', 'ball']:
            tdf = frame_df[frame_df['team'] == team_name]
            if tdf.empty:
                continue

            color = COLOR_MAP_HEX.get(team_name, '#8E8E93')
            size = 250 if team_name != 'ball' else 120
            marker = 'o' if team_name != 'ball' else '*'

            pitch.scatter(
                tdf['x'], tdf['y'],
                ax=ax_pitch,
                c=color,
                s=size,
                marker=marker,
                edgecolors='white',
                linewidth=1.5,
                zorder=5 if team_name != 'ball' else 6,
                label=team_name.upper()
            )

            # Annotate player numbers
            for _, row in tdf.iterrows():
                if team_name != 'ball':
                    pitch.annotate(
                        text=f"#{row['player_id']}",
                        xy=(row['x'], row['y']),
                        ax=ax_pitch,
                        ha='center', va='center',
                        color='white',
                        fontsize=9,
                        fontweight='bold',
                        zorder=7
                    )

        save_file = output_dir / f"validation_frame_{fid:04d}.png"
        plt.tight_layout()
        plt.savefig(save_file, dpi=120, bbox_inches='tight')
        plt.close(fig)
        validation_paths.append(str(save_file))
        print(f"Saved pitch validation image: {save_file}")

    return validation_paths


class PygameTacticalRenderer:
    """
    Pygame surface-based frame renderer.
    Draws pitches, team positions, ball, pass lines, shot highlights, and commentary overlays.
    """
    def __init__(self, pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH, scale=10, top_margin=60, bottom_margin=80):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.scale = scale
        self.top_margin = top_margin
        self.bottom_margin = bottom_margin

        self.pw = int(pitch_length * scale)
        self.ph = int(pitch_width * scale)
        self.width = self.pw + 40   # 20px padding left/right
        self.height = self.ph + top_margin + bottom_margin

        pygame.init()
        pygame.font.init()
        self.surface = pygame.Surface((self.width, self.height))

        self.font_title = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_body = pygame.font.SysFont("Arial", 16)
        self.font_player = pygame.font.SysFont("Arial", 13, bold=True)

    def pitch_to_screen(self, px, py):
        """Map (0..105, 0..68) pitch meters to Pygame screen coordinates."""
        sx = int(20 + px * self.scale)
        sy = int(self.top_margin + py * self.scale)
        return sx, sy

    def draw_pitch_lines(self):
        """Draw grass field and white boundary/penalty lines."""
        # Grass background
        self.surface.fill((30, 58, 30))

        # Pitch Boundary
        rect = (20, self.top_margin, self.pw, self.ph)
        pygame.draw.rect(self.surface, (255, 255, 255), rect, 3)

        # Halfway line & Center Circle
        mid_x = int(20 + (self.pw / 2.0))
        mid_y = int(self.top_margin + (self.ph / 2.0))
        pygame.draw.line(self.surface, (255, 255, 255), (mid_x, self.top_margin), (mid_x, self.top_margin + self.ph), 2)
        pygame.draw.circle(self.surface, (255, 255, 255), (mid_x, mid_y), int(9.15 * self.scale), 2)
        pygame.draw.circle(self.surface, (255, 255, 255), (mid_x, mid_y), 4)

        # Penalty Areas (16.5m x 40.32m)
        box_w = int(16.5 * self.scale)
        box_h = int(40.32 * self.scale)
        box_top = int(self.top_margin + ((self.pitch_width - 40.32) / 2.0) * self.scale)
        # Left Box
        pygame.draw.rect(self.surface, (255, 255, 255), (20, box_top, box_w, box_h), 2)
        # Right Box
        pygame.draw.rect(self.surface, (255, 255, 255), (20 + self.pw - box_w, box_top, box_w, box_h), 2)

    def render_frame(self, frame_id, frame_df, active_events=None, commentary_text=""):
        """Draw one full frame and return RGB numpy array."""
        self.draw_pitch_lines()

        # Header Title Banner
        pygame.draw.rect(self.surface, (15, 30, 15), (0, 0, self.width, self.top_margin))
        title_surf = self.font_title.render(f"2D Tactical Radar - Frame {frame_id}", True, (255, 255, 255))
        self.surface.blit(title_surf, (20, 15))

        # Render Players & Ball
        ball_pos = None
        for _, row in frame_df.iterrows():
            sx, sy = self.pitch_to_screen(row['x'], row['y'])
            team = row['team']

            if team == 'ball':
                ball_pos = (sx, sy)
                pygame.draw.circle(self.surface, COLOR_MAP['ball'], (sx, sy), 8)
                pygame.draw.circle(self.surface, (0, 0, 0), (sx, sy), 8, 1)
            else:
                color = COLOR_MAP.get(team, COLOR_MAP['unknown'])
                pygame.draw.circle(self.surface, color, (sx, sy), 13)
                pygame.draw.circle(self.surface, (255, 255, 255), (sx, sy), 13, 2)

                # Player Number Tag
                lbl = self.font_player.render(str(row['player_id']), True, (255, 255, 255))
                lw, lh = lbl.get_size()
                self.surface.blit(lbl, (sx - lw // 2, sy - lh // 2))

        # Event Highlight Overlays
        if active_events:
            for ev in active_events:
                ev_type = ev.get('type', '')
                ev_frame = ev.get('frame', -1)
                if abs(ev_frame - frame_id) <= 15:
                    if ev_type == 'pass':
                        # Highlight pass line if coordinates exist
                        p_from = frame_df[frame_df['track_id'] == ev.get('from')]
                        p_to = frame_df[frame_df['track_id'] == ev.get('to')]
                        if not p_from.empty and not p_to.empty:
                            x1, y1 = self.pitch_to_screen(p_from.iloc[0]['x'], p_from.iloc[0]['y'])
                            x2, y2 = self.pitch_to_screen(p_to.iloc[0]['x'], p_to.iloc[0]['y'])
                            pygame.draw.line(self.surface, (0, 255, 200), (x1, y1), (x2, y2), 3)

                    elif ev_type in ('shot', 'possible_goal', 'goal') and ball_pos:
                        # Flash shot ring around ball
                        pygame.draw.circle(self.surface, (255, 0, 0), ball_pos, 22, 3)

        # Footer Commentary Banner
        footer_y = self.top_margin + self.ph + 10
        pygame.draw.rect(self.surface, (15, 30, 15), (0, footer_y, self.width, self.bottom_margin))
        if commentary_text:
            comm_surf = self.font_body.render(f"Commentary: {commentary_text[:90]}", True, (255, 220, 100))
            self.surface.blit(comm_surf, (20, footer_y + 15))

        # Legend
        leg_x = self.width - 240
        pygame.draw.circle(self.surface, COLOR_MAP['team_a'], (leg_x, 30), 7)
        self.surface.blit(self.font_player.render("Team A", True, (255, 255, 255)), (leg_x + 12, 22))
        pygame.draw.circle(self.surface, COLOR_MAP['team_b'], (leg_x + 80, 30), 7)
        self.surface.blit(self.font_player.render("Team B", True, (255, 255, 255)), (leg_x + 92, 22))
        pygame.draw.circle(self.surface, COLOR_MAP['ball'], (leg_x + 160, 30), 5)
        self.surface.blit(self.font_player.render("Ball", True, (255, 255, 255)), (leg_x + 170, 22))

        # Convert to RGB numpy array
        view = pygame.surfarray.array3d(self.surface)
        return np.transpose(view, (1, 0, 2))


def export_tactical_video(df, events, commentary_text, output_mp4_path, fps=25.0,
                          pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH):
    """
    Iterates frame_id in strict order, renders Pygame surface, and writes MP4 via MoviePy.
    """
    ImageSequenceClip = None
    try:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    except Exception:
        try:
            import moviepy.editor as mpy
            ImageSequenceClip = mpy.ImageSequenceClip
        except Exception:
            ImageSequenceClip = None

    output_mp4_path = Path(output_mp4_path)
    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = PygameTacticalRenderer(pitch_length=pitch_length, pitch_width=pitch_width)
    frames_list = []

    available_frames = sorted(df['frame_id'].unique())
    if not available_frames:
        print("Error: No frames available in dataframe for video export.")
        return str(output_mp4_path)

    print(f"Rendering {len(available_frames)} tactical pitch frames with Pygame...")
    for fid in available_frames:
        frame_df = df[df['frame_id'] == fid]
        # Active events for current frame
        active_evs = [ev for ev in events if abs(ev.get('frame', -1) - fid) <= 15] if events else []
        rgb_frame = renderer.render_frame(fid, frame_df, active_events=active_evs, commentary_text=commentary_text)
        frames_list.append(rgb_frame)

    if ImageSequenceClip is not None:
        try:
            clip = ImageSequenceClip(frames_list, fps=fps)
            clip.write_videofile(str(output_mp4_path), codec='libx264', audio=False, logger=None)
            print(f"Exported tactical video with MoviePy to: {output_mp4_path}")
            return str(output_mp4_path)
        except Exception as e:
            print(f"MoviePy video export failed ({e}); falling back to OpenCV VideoWriter...")

    # Fallback OpenCV VideoWriter
    h, w, _ = frames_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_mp4_path), fourcc, fps, (w, h))
    for f in frames_list:
        bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()
    print(f"Exported tactical video with OpenCV to: {output_mp4_path}")
    return str(output_mp4_path)


def export_plotly_debug_html(df, events, output_html_path, pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH):
    """
    Generates an interactive Plotly HTML debug visualization with a frame slider.
    """
    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    available_frames = sorted(df['frame_id'].unique())
    if not available_frames:
        print("Error: Empty DataFrame passed to export_plotly_debug_html.")
        return str(output_html_path)

    # Pitch Outline Shapes
    shapes = [
        # Field Boundary
        dict(type="rect", x0=0, y0=0, x1=pitch_length, y1=pitch_width, line=dict(color="white", width=2), fillcolor="#1E3A1E"),
        # Halfway Line
        dict(type="line", x0=pitch_length / 2.0, y0=0, x1=pitch_length / 2.0, y1=pitch_width, line=dict(color="white", width=2)),
        # Center Circle
        dict(type="circle", x0=(pitch_length / 2.0) - 9.15, y0=(pitch_width / 2.0) - 9.15,
             x1=(pitch_length / 2.0) + 9.15, y1=(pitch_width / 2.0) + 9.15, line=dict(color="white", width=2)),
        # Left Penalty Box
        dict(type="rect", x0=0, y0=(pitch_width - 40.32) / 2.0, x1=16.5, y1=(pitch_width + 40.32) / 2.0, line=dict(color="white", width=2)),
        # Right Penalty Box
        dict(type="rect", x0=pitch_length - 16.5, y0=(pitch_width - 40.32) / 2.0, x1=pitch_length, y1=(pitch_width + 40.32) / 2.0, line=dict(color="white", width=2))
    ]

    # Create Initial Traces for Frame 0
    first_fid = available_frames[0]
    initial_df = df[df['frame_id'] == first_fid]

    fig = go.Figure()

    for team_name in ['team_a', 'team_b', 'referee', 'ball']:
        tdf = initial_df[initial_df['team'] == team_name]
        color = COLOR_MAP_HEX.get(team_name, '#8E8E93')
        size = 14 if team_name != 'ball' else 10

        fig.add_trace(go.Scatter(
            x=tdf['x'],
            y=tdf['y'],
            mode='markers+text',
            text=tdf['player_id'],
            textposition="top center",
            marker=dict(size=size, color=color, line=dict(width=1, color='white')),
            name=team_name.upper(),
            hoverinfo='text',
            hovertext=[f"Frame: {r['frame_id']}<br>Team: {r['team']}<br>Player: #{r['player_id']}<br>Pos: ({r['x']}m, {r['y']}m)" for _, r in tdf.iterrows()]
        ))

    # Build Frames for Animation Slider
    plotly_frames = []
    for fid in available_frames:
        frame_df = df[df['frame_id'] == fid]
        frame_data = []
        for team_name in ['team_a', 'team_b', 'referee', 'ball']:
            tdf = frame_df[frame_df['team'] == team_name]
            color = COLOR_MAP_HEX.get(team_name, '#8E8E93')
            size = 14 if team_name != 'ball' else 10
            frame_data.append(go.Scatter(
                x=tdf['x'],
                y=tdf['y'],
                mode='markers+text',
                text=tdf['player_id'],
                textposition="top center",
                marker=dict(size=size, color=color, line=dict(width=1, color='white')),
                name=team_name.upper(),
                hoverinfo='text',
                hovertext=[f"Frame: {r['frame_id']}<br>Team: {r['team']}<br>Player: #{r['player_id']}<br>Pos: ({r['x']}m, {r['y']}m)" for _, r in tdf.iterrows()]
            ))
        plotly_frames.append(go.Frame(data=frame_data, name=str(fid)))

    fig.frames = plotly_frames

    # Animation Slider Configuration
    slider_steps = []
    for fid in available_frames:
        step = dict(
            method="animate",
            args=[[str(fid)], dict(mode="immediate", frame=dict(duration=50, redraw=True), transition=dict(duration=0))],
            label=str(fid)
        )
        slider_steps.append(step)

    fig.update_layout(
        title="Tactical Radar Debug Artifact (Plotly Pitch)",
        xaxis=dict(range=[-5, pitch_length + 5], autorange=False, showgrid=False, zeroline=False),
        yaxis=dict(range=[-5, pitch_width + 5], autorange=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        shapes=shapes,
        sliders=[dict(active=0, yanchor="top", xanchor="left", currentvalue=dict(font=dict(size=14), prefix="Frame: ", visible=True, xanchor="right"), steps=slider_steps)],
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
                dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
            ]
        )],
        width=1000,
        height=680,
        paper_bgcolor="#121212",
        plot_bgcolor="#1E3A1E",
        font=dict(color="white")
    )

    fig.write_html(str(output_html_path))
    print(f"Exported Plotly HTML debug artifact to: {output_html_path}")
    return str(output_html_path)
