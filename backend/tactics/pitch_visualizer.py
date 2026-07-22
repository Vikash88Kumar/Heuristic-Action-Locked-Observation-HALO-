"""
Pitch visualization module using mplsoccer for football tactical analysis.
Creates beautiful 2D pitch diagrams with formations, player positions, and movements.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from mplsoccer import Pitch
import numpy as np


class PitchVisualizer:
    """Main class for visualizing football tactics on a pitch."""
    
    # Standard pitch dimensions (in meters)
    PITCH_LENGTH = 105
    PITCH_WIDTH = 68
    
    # Color schemes for teams
    TEAM_COLORS = {
        'home': '#1f77b4',      # Blue
        'away': '#ff7f0e',       # Orange
        'neutral': '#7f7f7f'     # Gray
    }
    
    # Formation positions (relative to pitch, 0-100 scale)
    FORMATION_POSITIONS = {
        '4-3-3': {
            'home': [
                (50, 10),   # GK
                (20, 15), (30, 20), (70, 20), (80, 15),  # Defenders
                (25, 50), (50, 55), (75, 50),  # Midfielders
                (20, 75), (50, 85), (80, 75),  # Forwards
            ],
            'away': [
                (50, 90),   # GK
                (20, 85), (30, 80), (70, 80), (80, 85),  # Defenders
                (25, 50), (50, 45), (75, 50),  # Midfielders
                (20, 25), (50, 15), (80, 25),  # Forwards
            ]
        },
        '4-2-3-1': {
            'home': [
                (50, 10),   # GK
                (20, 15), (30, 20), (70, 20), (80, 15),  # Defenders
                (30, 40), (70, 40),  # Defensive Midfielders
                (25, 60), (50, 65), (75, 60),  # Attacking Midfielders
                (50, 85),  # Striker
            ],
            'away': [
                (50, 90),   # GK
                (20, 85), (30, 80), (70, 80), (80, 85),  # Defenders
                (30, 60), (70, 60),  # Defensive Midfielders
                (25, 40), (50, 35), (75, 40),  # Attacking Midfielders
                (50, 15),  # Striker
            ]
        },
        '5-3-2': {
            'home': [
                (50, 10),   # GK
                (15, 20), (25, 15), (50, 12), (75, 15), (85, 20),  # Defenders
                (25, 50), (50, 55), (75, 50),  # Midfielders
                (30, 75), (70, 75),  # Forwards
            ],
            'away': [
                (50, 90),   # GK
                (15, 80), (25, 85), (50, 88), (75, 85), (85, 80),  # Defenders
                (25, 50), (50, 45), (75, 50),  # Midfielders
                (30, 25), (70, 25),  # Forwards
            ]
        }
    }
    
    def __init__(self, pitch_type='statsbomb', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the pitch visualizer.
        
        Args:
            pitch_type: Type of pitch ('statsbomb', 'wyscout', 'tracab', 'event')
            figsize: Figure size (width, height) in inches
        """
        self.pitch_type = pitch_type
        self.figsize = figsize
        
    def create_formation_diagram(
        self,
        formation: str,
        title: str = "Formation",
        save_path: Optional[Path] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a formation diagram on a pitch.
        
        Args:
            formation: Formation string (e.g., '4-3-3', '4-2-3-1')
            title: Diagram title
            save_path: Path to save the figure
            
        Returns:
            Tuple of (figure, axes)
        """
        if formation not in self.FORMATION_POSITIONS:
            raise ValueError(f"Formation {formation} not supported")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        pitch = Pitch(pitch_type=self.pitch_type, pitch_color='grass')
        pitch.draw(ax=ax)
        
        positions = self.FORMATION_POSITIONS[formation]
        
        # Draw home team (bottom)
        home_positions = positions['home']
        for idx, (x, y) in enumerate(home_positions):
            if idx == 0:  # Goalkeeper
                color = self.TEAM_COLORS['home']
                marker = 's'
                size = 200
            else:
                color = self.TEAM_COLORS['home']
                marker = 'o'
                size = 150
            
            ax.scatter(x, y, s=size, c=color, marker=marker, edgecolors='white', linewidth=2, zorder=5)
            ax.text(x, y, str(idx), ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=6)
        
        # Draw away team (top)
        away_positions = positions['away']
        for idx, (x, y) in enumerate(away_positions):
            if idx == 0:  # Goalkeeper
                color = self.TEAM_COLORS['away']
                marker = 's'
                size = 200
            else:
                color = self.TEAM_COLORS['away']
                marker = 'o'
                size = 150
            
            ax.scatter(x, y, s=size, c=color, marker=marker, edgecolors='white', linewidth=2, zorder=5)
            ax.text(x, y, str(idx), ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=6)
        
        ax.set_title(f"{title}\nFormation: {formation}", fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        home_patch = mpatches.Patch(color=self.TEAM_COLORS['home'], label='Home Team')
        away_patch = mpatches.Patch(color=self.TEAM_COLORS['away'], label='Away Team')
        ax.legend(handles=[home_patch, away_patch], loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax

    def create_formation_comparison(
        self,
        home_formation: str,
        away_formation: str,
        title: str = "Formation Comparison",
        save_path: Optional[Path] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a formation comparison diagram on a single pitch.
        
        Args:
            home_formation: Home team formation string
            away_formation: Away team formation string
            title: Diagram title
            save_path: Path to save the figure
            
        Returns:
            Tuple of (figure, axes)
        """
        if home_formation not in self.FORMATION_POSITIONS:
            home_formation = '4-3-3'
        if away_formation not in self.FORMATION_POSITIONS:
            away_formation = '4-3-3'
            
        fig, ax = plt.subplots(figsize=self.figsize)
        pitch = Pitch(pitch_type=self.pitch_type, pitch_color='grass')
        pitch.draw(ax=ax)
        
        # Draw home team (bottom) using home_formation
        home_positions = self.FORMATION_POSITIONS[home_formation]['home']
        for idx, (x, y) in enumerate(home_positions):
            if idx == 0:  # Goalkeeper
                color = self.TEAM_COLORS['home']
                marker = 's'
                size = 200
            else:
                color = self.TEAM_COLORS['home']
                marker = 'o'
                size = 150
            
            ax.scatter(x, y, s=size, c=color, marker=marker, edgecolors='white', linewidth=2, zorder=5)
            ax.text(x, y, str(idx), ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=6)
            
        # Draw away team (top) using away_formation
        away_positions = self.FORMATION_POSITIONS[away_formation]['away']
        for idx, (x, y) in enumerate(away_positions):
            if idx == 0:  # Goalkeeper
                color = self.TEAM_COLORS['away']
                marker = 's'
                size = 200
            else:
                color = self.TEAM_COLORS['away']
                marker = 'o'
                size = 150
            
            ax.scatter(x, y, s=size, c=color, marker=marker, edgecolors='white', linewidth=2, zorder=5)
            ax.text(x, y, str(idx), ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=6)
            
        ax.set_title(f"{title}\nHome: {home_formation} | Away: {away_formation}", fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        home_patch = mpatches.Patch(color=self.TEAM_COLORS['home'], label=f'Home ({home_formation})')
        away_patch = mpatches.Patch(color=self.TEAM_COLORS['away'], label=f'Away ({away_formation})')
        ax.legend(handles=[home_patch, away_patch], loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig, ax
    
    def create_tactical_analysis(
        self,
        tactical_data: Dict,
        save_path: Optional[Path] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create a 4-subplot tactical analysis.
        
        Args:
            tactical_data: Dictionary with tactical analysis
            save_path: Path to save the figure
            
        Returns:
            Tuple of (figure, axes array)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Build-up play
        pitch1 = Pitch(pitch_type=self.pitch_type, pitch_color='grass')
        pitch1.draw(ax=axes[0, 0])
        axes[0, 0].set_title('Build-up Play', fontweight='bold')
        
        # Add text description
        build_up = tactical_data.get('build_up_style', 'N/A')
        axes[0, 0].text(0.5, 0.05, f"Style: {build_up}", ha='center', transform=axes[0, 0].transAxes)
        
        # Pressing
        pitch2 = Pitch(pitch_type=self.pitch_type, pitch_color='grass')
        pitch2.draw(ax=axes[0, 1])
        axes[0, 1].set_title('Pressing', fontweight='bold')
        pressing = tactical_data.get('pressing_style', 'N/A')
        axes[0, 1].text(0.5, 0.05, f"Style: {pressing}", ha='center', transform=axes[0, 1].transAxes)
        
        # Attacking Patterns
        pitch3 = Pitch(pitch_type=self.pitch_type, pitch_color='grass')
        pitch3.draw(ax=axes[1, 0])
        axes[1, 0].set_title('Attacking Patterns', fontweight='bold')
        attack = tactical_data.get('attacking_patterns', 'N/A')
        if isinstance(attack, list):
            attack = ', '.join(attack[:2])
        axes[1, 0].text(0.5, 0.05, f"Patterns: {attack}", ha='center', transform=axes[1, 0].transAxes)
        
        # Defensive Shape
        pitch4 = Pitch(pitch_type=self.pitch_type, pitch_color='grass')
        pitch4.draw(ax=axes[1, 1])
        axes[1, 1].set_title('Defensive Shape', fontweight='bold')
        defense = tactical_data.get('defensive_shape', 'N/A')
        axes[1, 1].text(0.5, 0.05, f"Shape: {defense}", ha='center', transform=axes[1, 1].transAxes)
        
        fig.suptitle('Tactical Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, axes
    
    def create_passing_network(
        self,
        passes: List[Dict],
        title: str = "Passing Network",
        save_path: Optional[Path] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a passing network visualization.
        
        Args:
            passes: List of pass dictionaries
            title: Diagram title
            save_path: Path to save the figure
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        pitch = Pitch(pitch_type=self.pitch_type, pitch_color='grass')
        pitch.draw(ax=ax)
        
        if not passes:
            return fig, ax
        
        # Aggregate pass data
        pass_counts = {}
        pass_connections = {}
        
        for pass_data in passes:
            team = pass_data.get('team', 'home')
            from_pos = (pass_data['from_x'], pass_data['from_y'])
            to_pos = (pass_data['to_x'], pass_data['to_y'])
            
            # Count passes from each position
            if from_pos not in pass_counts:
                pass_counts[from_pos] = {'count': 0, 'team': team}
            pass_counts[from_pos]['count'] += 1
            
            # Count pass connections
            connection = (from_pos, to_pos)
            if connection not in pass_connections:
                pass_connections[connection] = {'count': 0, 'team': team}
            pass_connections[connection]['count'] += 1
        
        # Draw pass connections as arrows
        for (from_pos, to_pos), data in pass_connections.items():
            team = data['team']
            count = data['count']
            color = self.TEAM_COLORS.get(team, self.TEAM_COLORS['neutral'])
            
            # Arrow width proportional to number of passes
            linewidth = max(0.5, min(3, count / 2))
            alpha = min(0.8, count / 10)
            
            arrow = FancyArrowPatch(
                from_pos, to_pos,
                arrowstyle='->', mutation_scale=15, linewidth=linewidth,
                color=color, alpha=alpha, zorder=2
            )
            ax.add_patch(arrow)
        
        # Draw player nodes
        for pos, data in pass_counts.items():
            team = data['team']
            count = data['count']
            color = self.TEAM_COLORS.get(team, self.TEAM_COLORS['neutral'])
            
            # Node size proportional to number of passes
            size = max(100, min(500, count * 20))
            
            ax.scatter(pos[0], pos[1], s=size, c=color, edgecolors='white', linewidth=2, zorder=5, alpha=0.7)
            ax.text(pos[0], pos[1], str(count), ha='center', va='center', fontsize=10, fontweight='bold', zorder=6)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def create_movement_heatmap(
        self,
        movements: List[Tuple[float, float]],
        team: str = 'home',
        title: str = "Player Movement Heatmap",
        save_path: Optional[Path] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a heatmap of player movements on the pitch.
        
        Args:
            movements: List of (x, y) coordinates
            team: Team identifier ('home' or 'away')
            title: Diagram title
            save_path: Path to save the figure
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        pitch = Pitch(pitch_type=self.pitch_type, pitch_color='grass')
        pitch.draw(ax=ax)
        
        # 2D histogram for heatmap
        if movements:
            x_coords = [m[0] for m in movements]
            y_coords = [m[1] for m in movements]
            
            heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=10, range=[[0, 100], [0, 100]])
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='YlOrRd', alpha=0.6, aspect='auto')
            plt.colorbar(im, ax=ax, label='Movement Density')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
