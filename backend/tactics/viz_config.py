"""
Visualization Configuration and Utilities

Provides configuration options and helper utilities for advanced customization.
"""

from typing import Dict, Tuple


class VisualizationConfig:
    """Configuration class for visualization customization."""
    
    # ========== COLOR SCHEMES ==========
    COLOR_SCHEMES = {
        'default': {
            'home': '#1f77b4',      # Blue
            'away': '#ff7f0e',      # Orange
            'neutral': '#7f7f7f',   # Gray
        },
        'classic': {
            'home': '#000000',      # Black
            'away': '#FFFFFF',      # White
            'neutral': '#888888',
        },
        'vivid': {
            'home': '#E74C3C',      # Red
            'away': '#3498DB',      # Blue
            'neutral': '#95A5A6',
        },
        'autumn': {
            'home': '#D35400',      # Orange
            'away': '#8E44AD',      # Purple
            'neutral': '#BDC3C7',
        },
        'nature': {
            'home': '#27AE60',      # Green
            'away': '#F39C12',      # Gold
            'neutral': '#95A5A6',
        }
    }
    
    # ========== PITCH TYPES ==========
    PITCH_TYPES = {
        'statsbomb': 'StatsBomb (recommended)',
        'wyscout': 'Wyscout',
        'tracab': 'Tracab',
        'event': 'Event',
    }
    
    # ========== FORMATIONS ==========
    SUPPORTED_FORMATIONS = {
        '4-3-3': 'Classic balanced formation',
        '4-2-3-1': 'Defensive midfielder formation',
        '5-3-2': 'Wing-back heavy formation',
        '3-5-2': 'Wing-back heavy (3 at back)',
        '4-4-2': 'Traditional formation',
    }
    
    # ========== ANIMATION SETTINGS ==========
    ANIMATION_DEFAULTS = {
        'attacking_frames': 30,
        'defensive_frames': 20,
        'fps': 8,
        'figsize': (12, 8),
        'dpi': 300,
    }
    
    # ========== PRESSING ZONES ==========
    PRESSING_ZONES = {
        'high_press': {
            'description': 'Aggressive pressing in final third',
            'x_range': (70, 105),
            'color': 'red',
            'alpha': 0.2,
        },
        'medium_press': {
            'description': 'Moderate engagement in midfield',
            'x_range': (45, 75),
            'color': 'yellow',
            'alpha': 0.2,
        },
        'low_block': {
            'description': 'Deep defensive positioning',
            'x_range': (0, 40),
            'color': 'blue',
            'alpha': 0.2,
        }
    }
    
    def __init__(self, color_scheme: str = 'default'):
        """
        Initialize configuration.
        
        Args:
            color_scheme: Name of color scheme to use
        """
        if color_scheme not in self.COLOR_SCHEMES:
            raise ValueError(f"Unknown color scheme: {color_scheme}")
        
        self.current_scheme = color_scheme
        self.colors = self.COLOR_SCHEMES[color_scheme].copy()
    
    def set_color_scheme(self, scheme: str) -> None:
        """Change color scheme."""
        if scheme not in self.COLOR_SCHEMES:
            raise ValueError(f"Unknown color scheme: {scheme}")
        self.current_scheme = scheme
        self.colors = self.COLOR_SCHEMES[scheme].copy()
    
    def set_team_colors(self, home: str, away: str) -> None:
        """Set custom team colors."""
        self.colors['home'] = home
        self.colors['away'] = away
    
    def list_color_schemes(self) -> Dict[str, Dict[str, str]]:
        """List all available color schemes."""
        return self.COLOR_SCHEMES.copy()
    
    def get_pitch_types(self) -> Dict[str, str]:
        """Get available pitch types."""
        return self.PITCH_TYPES.copy()


class TacticalPatterns:
    """Predefined tactical patterns for reference."""
    
    DEFENSIVE_PATTERNS = {
        'high_press': {
            'description': 'Aggressive press to win ball high up pitch',
            'pressing_style': 'Aggressive high press in final third',
            'defensive_shape': 'High block with man-to-man marking',
            'transition_style': 'Quick counter-attacks after winning ball',
        },
        'medium_press': {
            'description': 'Balanced pressing in middle third',
            'pressing_style': 'Moderate press in midfield',
            'defensive_shape': 'Mid-block with zonal coverage',
            'transition_style': 'Structured counter-attacks',
        },
        'low_block': {
            'description': 'Deep defensive organization',
            'pressing_style': 'Minimal pressing, allow possession',
            'defensive_shape': 'Deep compact block',
            'transition_style': 'Counter-attacks from deep',
        },
    }
    
    ATTACKING_PATTERNS = {
        'possession_based': {
            'description': 'Patient buildup with possession focus',
            'build_up_style': 'Short passes, central progression',
            'attacking_patterns': 'Gradual width creation',
            'transition_style': 'Possession-oriented transitions',
        },
        'direct': {
            'description': 'Direct long-ball play',
            'build_up_style': 'Long balls from defense',
            'attacking_patterns': 'Direct vertical passes',
            'transition_style': 'Quick transitions with long balls',
        },
        'wing_focused': {
            'description': 'Attacks focused on flanks',
            'build_up_style': 'Build to fullbacks',
            'attacking_patterns': 'Frequent crosses and wing plays',
            'transition_style': 'Quick transitions through wings',
        },
        'through_ball': {
            'description': 'Penetrating passes between lines',
            'build_up_style': 'Buildup to midfielders',
            'attacking_patterns': 'Through balls to forwards',
            'transition_style': 'Direct penetrating transitions',
        },
    }
    
    @staticmethod
    def get_defensive_pattern(pattern: str) -> Dict:
        """Get a defensive pattern template."""
        return TacticalPatterns.DEFENSIVE_PATTERNS.get(pattern, {})
    
    @staticmethod
    def get_attacking_pattern(pattern: str) -> Dict:
        """Get an attacking pattern template."""
        return TacticalPatterns.ATTACKING_PATTERNS.get(pattern, {})
    
    @staticmethod
    def combine_patterns(defensive: str, attacking: str) -> Dict:
        """Combine defensive and attacking patterns."""
        result = {
            'defensive': TacticalPatterns.get_defensive_pattern(defensive),
            'attacking': TacticalPatterns.get_attacking_pattern(attacking),
        }
        return result


class FormationAnalyzer:
    """Utilities for formation analysis."""
    
    # Formation characteristics
    FORMATION_CHARACTERISTICS = {
        '4-3-3': {
            'balance': 'Balanced',
            'defense': 'Solid',
            'midfield': 'Controlled',
            'attack': 'Flexible',
            'best_for': 'Versatility',
            'weakness': 'Can be exposed on wings',
        },
        '4-2-3-1': {
            'balance': 'Defensive',
            'defense': 'Very solid',
            'midfield': 'Protected',
            'attack': 'Clinical',
            'best_for': 'Defensive stability',
            'weakness': 'May lack creative midfield',
        },
        '5-3-2': {
            'balance': 'Defensive',
            'defense': 'Very strong',
            'midfield': 'Controlled',
            'attack': 'Direct',
            'best_for': 'Defensive solidity',
            'weakness': 'Limited attacking width',
        },
        '3-5-2': {
            'balance': 'Attacking',
            'defense': 'Vulnerable',
            'midfield': 'Dominant',
            'attack': 'Strong',
            'best_for': 'Midfield dominance',
            'weakness': 'Defensive fragility',
        },
        '4-4-2': {
            'balance': 'Balanced',
            'defense': 'Solid',
            'midfield': 'Balanced',
            'attack': 'Direct',
            'best_for': 'Traditional play',
            'weakness': 'Can lack midfield control',
        },
    }
    
    @staticmethod
    def get_formation_info(formation: str) -> Dict:
        """Get detailed information about a formation."""
        return FormationAnalyzer.FORMATION_CHARACTERISTICS.get(formation, {})
    
    @staticmethod
    def compare_formations(formation1: str, formation2: str) -> Dict:
        """Compare two formations."""
        info1 = FormationAnalyzer.get_formation_info(formation1)
        info2 = FormationAnalyzer.get_formation_info(formation2)
        
        return {
            'formation1': {
                'name': formation1,
                'info': info1,
            },
            'formation2': {
                'name': formation2,
                'info': info2,
            },
            'matchup_analysis': FormationAnalyzer._analyze_matchup(formation1, formation2)
        }
    
    @staticmethod
    def _analyze_matchup(formation1: str, formation2: str) -> str:
        """Analyze how formations match up."""
        matchups = {
            ('4-3-3', '4-2-3-1'): 'Balanced vs Defensive - Formation 1 has more attacking potential',
            ('4-2-3-1', '4-3-3'): 'Defensive vs Balanced - Formation 2 has more attacking potential',
            ('5-3-2', '4-3-3'): 'Defensive vs Balanced - Formation 2 has more attacking potential',
            ('3-5-2', '4-3-3'): 'Attacking vs Balanced - Formation 1 may dominate midfield',
        }
        
        key = tuple(sorted([formation1, formation2]))
        return matchups.get(key, 'Different tactical approaches')


# ========== HELPER FUNCTIONS ==========

def create_custom_color_scheme(
    home: str,
    away: str,
    neutral: str = '#7f7f7f'
) -> Dict[str, str]:
    """Create a custom color scheme."""
    return {
        'home': home,
        'away': away,
        'neutral': neutral,
    }


def recommend_formations_for_opposition(
    opponent_formation: str,
) -> Tuple[str, str, str]:
    """Recommend formations to counter opponent."""
    recommendations = {
        '4-3-3': ('4-2-3-1', '5-3-2', '4-3-3'),
        '4-2-3-1': ('4-3-3', '3-5-2', '4-4-2'),
        '5-3-2': ('4-3-3', '3-5-2', '4-4-2'),
        '3-5-2': ('4-2-3-1', '5-3-2', '4-3-3'),
        '4-4-2': ('4-3-3', '4-2-3-1', '3-5-2'),
    }
    
    return recommendations.get(opponent_formation, ('4-3-3', '4-2-3-1', '5-3-2'))


def get_tactical_advantage(
    formation1: str,
    formation2: str,
) -> str:
    """Determine potential tactical advantage."""
    advantages = {
        ('4-3-3', '4-4-2'): 'Formation 1 has midfield superiority',
        ('4-2-3-1', '3-5-2'): 'Formation 1 has defensive stability',
        ('5-3-2', '4-3-3'): 'Formation 1 is more defensively solid',
        ('3-5-2', '4-4-2'): 'Formation 1 may dominate midfield',
    }
    
    key = tuple(sorted([formation1, formation2]))
    reversed_key = (formation2, formation1)
    
    if key in advantages:
        return advantages[key]
    elif reversed_key in advantages:
        return advantages[reversed_key].replace('Formation 1', 'Formation 2')
    
    return 'Similar tactical profiles'


# ========== EXAMPLE USAGE ==========

if __name__ == "__main__":
    # Example: Configure visualizations
    print("=== Visualization Configuration Examples ===\n")
    
    # Use default color scheme
    config = VisualizationConfig('default')
    print(f"Color Scheme (default): {config.colors}")
    
    # Switch to vivid scheme
    config.set_color_scheme('vivid')
    print(f"Color Scheme (vivid): {config.colors}")
    
    # Custom colors
    config.set_team_colors('#E74C3C', '#3498DB')
    print(f"Custom colors: {config.colors}\n")
    
    # Formation analysis
    print("=== Formation Analysis ===\n")
    print("4-3-3 Info:", FormationAnalyzer.get_formation_info('4-3-3'))
    print("\nRecommended vs 4-2-3-1:", recommend_formations_for_opposition('4-2-3-1'))
    print("\nTactical Advantage (4-3-3 vs 5-3-2):", get_tactical_advantage('4-3-3', '5-3-2'))
    
    # Tactical patterns
    print("\n=== Tactical Patterns ===\n")
    print("High Press Pattern:", TacticalPatterns.get_defensive_pattern('high_press'))
    print("\nWing-Focused Attack:", TacticalPatterns.get_attacking_pattern('wing_focused'))
