"""
Tactical mapping and simulation utilities for football analytics.
Converts tactical data into visual representations and creates movement simulations.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch, Circle, Wedge
from mplsoccer import Pitch
from scipy.interpolate import interp1d


@dataclass
class Player:
    """Represents a player on the pitch."""
    player_id: int
    name: str
    x: float
    y: float
    team: str
    position: str
    jersey_number: int = 0


@dataclass
class Movement:
    """Represents a player movement or action."""
    player_id: int
    start_pos: Tuple[float, float]
    end_pos: Tuple[float, float]
    action_type: str  # 'pass', 'dribble', 'run', 'press'
    timestamp: float = 0
    success: bool = True


class TacticalMapper:
    """Maps tactical data from analysis to playable coordinates."""
    
    PITCH_LENGTH = 105
    PITCH_WIDTH = 68
    
    # Position categories and their typical zones
    POSITION_ZONES = {
        'goalkeeper': {'x': (5, 15), 'y': (20, 48)},
        'defender': {'x': (10, 35), 'y': (0, 68)},
        'left_back': {'x': (10, 35), 'y': (0, 20)},
        'right_back': {'x': (10, 35), 'y': (48, 68)},
        'centre_back': {'x': (10, 35), 'y': (25, 43)},
        'midfielder': {'x': (35, 70), 'y': (10, 58)},
        'left_mid': {'x': (35, 70), 'y': (0, 25)},
        'right_mid': {'x': (35, 70), 'y': (43, 68)},
        'central_mid': {'x': (35, 70), 'y': (25, 43)},
        'forward': {'x': (70, 105), 'y': (15, 53)},
        'left_wing': {'x': (65, 105), 'y': (0, 25)},
        'right_wing': {'x': (65, 105), 'y': (43, 68)},
        'striker': {'x': (80, 105), 'y': (30, 38)},
    }
    
    # Pressing zones (defensive positions)
    PRESSING_ZONES = {
        'high_press': {'x': (70, 105), 'y': (0, 68)},
        'medium_press': {'x': (45, 75), 'y': (0, 68)},
        'low_block': {'x': (0, 40), 'y': (0, 68)},
    }
    
    def __init__(self):
        pass
    
    def parse_formation_to_positions(
        self,
        formation: str,
        team: str = 'home'
    ) -> List[Tuple[float, float]]:
        """
        Parse a formation string into player positions.
        
        Args:
            formation: Formation string like '4-3-3'
            team: 'home' or 'away'
            
        Returns:
            List of (x, y) positions
        """
        # Convert formation string to numbers
        parts = formation.split('-')
        try:
            if len(parts) == 4:
                defenders = int(parts[0])
                midfielders = int(parts[1]) + int(parts[2])
                forwards = int(parts[3])
            elif len(parts) == 3:
                defenders, midfielders, forwards = map(int, parts)
            elif len(parts) == 5:
                defenders = int(parts[0])
                midfielders = int(parts[1]) + int(parts[2]) + int(parts[3])
                forwards = int(parts[4])
            else:
                defenders, midfielders, forwards = 4, 3, 3
        except Exception:
            defenders, midfielders, forwards = 4, 3, 3
        
        positions = []
        
        # Add goalkeeper
        if team == 'home':
            positions.append((7, 34))
        else:
            positions.append((98, 34))
        
        # Add defenders
        defender_spacing = 68 / (defenders + 1)
        for i in range(1, defenders + 1):
            y = i * defender_spacing
            x = 20 if team == 'home' else 85
            positions.append((x, y))
        
        # Add midfielders
        mid_spacing = 68 / (midfielders + 1)
        for i in range(1, midfielders + 1):
            y = i * mid_spacing
            x = 50 if team == 'home' else 55
            positions.append((x, y))
        
        # Add forwards
        forward_spacing = 68 / (forwards + 1)
        for i in range(1, forwards + 1):
            y = i * forward_spacing
            x = 80 if team == 'home' else 25
            positions.append((x, y))
        
        return positions
    
    def map_pressing_style(self, pressing_style: str) -> str:
        """Map pressing style description to pressing zone."""
        pressing_lower = pressing_style.lower()
        
        if 'high' in pressing_lower or 'aggressive' in pressing_lower:
            return 'high_press'
        elif 'medium' in pressing_lower or 'mid' in pressing_lower:
            return 'medium_press'
        else:
            return 'low_block'
    
    def generate_attacking_pattern(
        self,
        attacking_style: str,
        start_pos: Tuple[float, float],
        num_touches: int = 5
    ) -> List[Tuple[float, float]]:
        """
        Generate attacking movement pattern based on style.
        
        Args:
            attacking_style: Description of attacking pattern
            start_pos: Starting position (x, y)
            num_touches: Number of touches in sequence
            
        Returns:
            List of positions showing the attack progression
        """
        pattern = [start_pos]
        x, y = start_pos
        
        style_lower = attacking_style.lower()
        
        if 'wing' in style_lower or 'flank' in style_lower:
            # Lateral movement along the wing
            for i in range(1, num_touches):
                x += 5  # Progress forward
                y += np.random.uniform(-8, 8)  # Lateral movement
                y = np.clip(y, 0, 68)
                pattern.append((x, y))
        
        elif 'through' in style_lower or 'penetrating' in style_lower:
            # Direct penetration
            for i in range(1, num_touches):
                x += 8
                y += np.random.uniform(-3, 3)
                x = np.clip(x, 0, 105)
                pattern.append((x, y))
        
        elif 'width' in style_lower or 'cross' in style_lower:
            # Movement to create crossing opportunities
            for i in range(1, num_touches):
                if i < num_touches // 2:
                    x += 3
                    y += 10 if i % 2 == 0 else -10
                else:
                    x += 5
                    y += np.random.uniform(-2, 2)
                y = np.clip(y, 0, 68)
                pattern.append((x, y))
        
        else:
            # Default: forward progression
            for i in range(1, num_touches):
                x += 4
                x = np.clip(x, 0, 105)
                pattern.append((x, y))
        
        return pattern
    
    def extract_tactical_events(
        self,
        tactical_data: Dict,
        team: str = 'home'
    ) -> Dict[str, any]:
        """
        Extract tactical events and patterns from analysis data.
        
        Args:
            tactical_data: Tactical analysis dictionary
            team: 'home' or 'away'
            
        Returns:
            Dictionary of tactical events
        """
        events = {
            'formation': tactical_data.get('formation', '4-3-3'),
            'pressing': self.map_pressing_style(tactical_data.get('pressing_style', 'Unknown')),
            'build_up': tactical_data.get('build_up_style', 'Short passes'),
            'attacking': tactical_data.get('attacking_patterns', 'Unknown'),
            'transition': tactical_data.get('transition_style', 'Unknown'),
            'strength': tactical_data.get('strengths', []),
            'weakness': tactical_data.get('weaknesses', []),
        }
        return events


class MovementSimulator:
    """Creates simulations and animations of tactical movements."""
    
    def __init__(self, pitch_type='statsbomb'):
        self.pitch_type = pitch_type
        self.mapper = TacticalMapper()
    
    def simulate_attacking_sequence(
        self,
        tactical_data: Dict,
        num_frames: int = 30
    ) -> List[List[Tuple[float, float]]]:
        """
        Simulate an attacking sequence based on tactical data.
        
        Args:
            tactical_data: Tactical analysis dictionary
            num_frames: Number of animation frames
            
        Returns:
            List of position lists for each frame
        """
        formation = tactical_data.get('formation', '4-3-3')
        attacking_style = tactical_data.get('attacking_patterns', 'Unknown')
        
        # Get initial positions
        initial_positions = self.mapper.parse_formation_to_positions(formation, 'home')
        
        # Simulate movement
        frames = []
        
        for frame in range(num_frames):
            frame_positions = []
            progress = frame / num_frames
            
            for idx, pos in enumerate(initial_positions):
                x, y = pos
                
                if idx == 0:  # Goalkeeper stays still
                    frame_positions.append((x, y))
                elif idx <= 4:  # Defenders move slightly
                    move_x = np.sin(progress * np.pi) * 5
                    move_y = np.cos(progress * np.pi * 2) * 3
                    frame_positions.append((x + move_x, np.clip(y + move_y, 0, 68)))
                else:  # Attackers move forward
                    move_x = progress * 20
                    move_y = np.sin(progress * np.pi * 3) * 5
                    x_new = np.clip(x + move_x, 0, 105)
                    y_new = np.clip(y + move_y, 0, 68)
                    frame_positions.append((x_new, y_new))
            
            frames.append(frame_positions)
        
        return frames
    
    def simulate_defensive_shape(
        self,
        pressing_zone: str,
        formation: str,
        num_frames: int = 20
    ) -> List[List[Tuple[float, float]]]:
        """
        Simulate defensive shape adjustments.
        
        Args:
            pressing_zone: 'high_press', 'medium_press', or 'low_block'
            formation: Formation string
            num_frames: Number of animation frames
            
        Returns:
            List of position lists for each frame
        """
        initial_positions = self.mapper.parse_formation_to_positions(formation, 'home')
        frames = []
        
        for frame in range(num_frames):
            frame_positions = []
            progress = frame / num_frames
            
            if pressing_zone == 'high_press':
                # Move forward aggressively
                adjustment = progress * 15
            elif pressing_zone == 'medium_press':
                # Moderate advancement
                adjustment = progress * 8
            else:  # low_block
                # Stay deep, defensive shape
                adjustment = -progress * 5
            
            for idx, (x, y) in enumerate(initial_positions):
                if idx == 0:  # Goalkeeper
                    frame_positions.append((x, y))
                elif idx <= 4:  # Defenders
                    x_new = np.clip(x + adjustment, 0, 105)
                    frame_positions.append((x_new, y))
                else:
                    # Midfielders and forwards follow
                    frame_positions.append((x, y))
            
            frames.append(frame_positions)
        
        return frames
    
    def create_animation(
        self,
        frames: List[List[Tuple[float, float]]],
        title: str = "Tactical Simulation",
        save_path: Optional[Path] = None,
        fps: int = 10
    ) -> plt.Figure:
        """
        Create an animated visualization of tactical movements.
        
        Args:
            frames: List of position lists for each frame
            title: Animation title
            save_path: Path to save the animation
            fps: Frames per second
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        pitch = Pitch(pitch_type=self.pitch_type, pitch_color='grass')
        pitch.draw(ax=ax)
        
        scatter = ax.scatter([], [], s=200, c='#1f77b4', edgecolors='white', linewidth=2, zorder=5)
        
        def update(frame_idx):
            if frame_idx < len(frames):
                positions = frames[frame_idx]
                x_coords = [p[0] for p in positions]
                y_coords = [p[1] for p in positions]
                scatter.set_offsets(np.c_[x_coords, y_coords])
                ax.set_title(f"{title} - Frame {frame_idx + 1}/{len(frames)}", fontsize=12, fontweight='bold')
            return scatter,
        
        anim = FuncAnimation(fig, update, frames=len(frames), interval=1000//fps, blit=True)
        
        if save_path:
            try:
                anim.save(save_path, writer='pillow', fps=fps)
            except Exception as e:
                print(f"Warning: Could not save animation: {e}")
        
        return fig
    
    def create_tactical_comparison(
        self,
        home_data: Dict,
        away_data: Dict,
        save_path: Optional[Path] = None
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Create a detailed tactical comparison visualization.
        
        Args:
            home_data: Home team tactical data
            away_data: Away team tactical data
            save_path: Path to save the figure
            
        Returns:
            Tuple of (figure, axes list)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Home Formation
        pitch1 = Pitch(pitch_type=self.pitch_type, pitch_color='grass')
        pitch1.draw(ax=axes[0, 0])
        home_formation = home_data.get('formation', '4-3-3')
        home_pos = self.mapper.parse_formation_to_positions(home_formation, 'home')
        x_home = [p[0] for p in home_pos]
        y_home = [p[1] for p in home_pos]
        axes[0, 0].scatter(x_home, y_home, s=200, c='#1f77b4', marker='o', edgecolors='white', linewidth=2, zorder=5)
        axes[0, 0].set_title(f"Home Formation: {home_formation}", fontsize=12, fontweight='bold')
        
        # Away Formation
        pitch2 = Pitch(pitch_type=self.pitch_type, pitch_color='grass')
        pitch2.draw(ax=axes[0, 1])
        away_formation = away_data.get('formation', '4-3-3')
        away_pos = self.mapper.parse_formation_to_positions(away_formation, 'away')
        x_away = [p[0] for p in away_pos]
        y_away = [p[1] for p in away_pos]
        axes[0, 1].scatter(x_away, y_away, s=200, c='#ff7f0e', marker='o', edgecolors='white', linewidth=2, zorder=5)
        axes[0, 1].set_title(f"Away Formation: {away_formation}", fontsize=12, fontweight='bold')
        
        # Tactical Summary
        pitch3 = Pitch(pitch_type=self.pitch_type, pitch_color='grass')
        pitch3.draw(ax=axes[1, 0])
        
        home_tactics = f"Build-up: {home_data.get('build_up_style', 'Unknown')}\n"
        home_tactics += f"Pressing: {home_data.get('pressing_style', 'Unknown')}"
        
        axes[1, 0].text(50, 50, home_tactics, ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), wrap=True)
        axes[1, 0].set_title("Home Tactics", fontsize=12, fontweight='bold')
        axes[1, 0].set_xlim(0, 100)
        axes[1, 0].set_ylim(0, 100)
        axes[1, 0].axis('off')
        
        # Away Tactics Summary
        pitch4 = Pitch(pitch_type=self.pitch_type, pitch_color='grass')
        pitch4.draw(ax=axes[1, 1])
        
        away_tactics = f"Build-up: {away_data.get('build_up_style', 'Unknown')}\n"
        away_tactics += f"Pressing: {away_data.get('pressing_style', 'Unknown')}"
        
        axes[1, 1].text(50, 50, away_tactics, ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8), wrap=True)
        axes[1, 1].set_title("Away Tactics", fontsize=12, fontweight='bold')
        axes[1, 1].set_xlim(0, 100)
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, axes
