"""
Visualization pipeline that generates all tactical diagrams and simulations from analysis data.
"""

import json
from pathlib import Path
from typing import Dict, Optional, List
import logging

from .pitch_visualizer import PitchVisualizer
from .tactical_mapper import TacticalMapper, MovementSimulator
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class TacticalVisualizationPipeline:
    """Main pipeline for generating tactical visualizations from analysis data."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the visualization pipeline.
        
        Args:
            output_dir: Directory to save visualizations (default: ./tactical_viz)
        """
        self.output_dir = output_dir or Path(__file__).parent / "tactical_viz"
        self.output_dir.mkdir(exist_ok=True)
        
        self.visualizer = PitchVisualizer()
        self.mapper = TacticalMapper()
        self.simulator = MovementSimulator()
        
        logger.info(f"Visualization pipeline initialized. Output dir: {self.output_dir}")
    
    def process_single_analysis(
        self,
        tactical_data: Dict,
        video_name: str,
        generate_animations: bool = True
    ) -> Dict[str, Path]:
        """
        Process a single tactical analysis and generate visualizations.
        
        Args:
            tactical_data: Dictionary from tactical analysis JSON
            video_name: Name of the video for file naming
            generate_animations: Whether to generate animated simulations
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        output_files = {}
        safe_name = video_name.replace(' ', '_').replace(':', '')
        
        try:
            # 1. Create formation diagram
            formation = tactical_data.get('formation', '4-3-3')
            fig, ax = self.visualizer.create_formation_diagram(
                formation=formation,
                title=f"Formation - {video_name}",
                save_path=self.output_dir / f"{safe_name}_formation.png"
            )
            plt.close(fig)
            output_files['formation'] = self.output_dir / f"{safe_name}_formation.png"
            logger.info(f"Generated formation diagram: {output_files['formation']}")
            
            # 2. Create tactical analysis (4-subplot figure)
            fig, axes = self.visualizer.create_tactical_analysis(
                tactical_data=tactical_data,
                save_path=self.output_dir / f"{safe_name}_tactical_analysis.png"
            )
            plt.close(fig)
            output_files['tactical_analysis'] = self.output_dir / f"{safe_name}_tactical_analysis.png"
            logger.info(f"Generated tactical analysis: {output_files['tactical_analysis']}")
            
            # 3. Create defensive shape visualization
            pressing_style = tactical_data.get('pressing_style', 'Unknown')
            pressing_zone = self.mapper.map_pressing_style(pressing_style)
            defensive_shape = tactical_data.get('defensive_shape', f"Pressing: {pressing_style}")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            from mplsoccer import Pitch
            pitch = Pitch(pitch_type='statsbomb', pitch_color='grass')
            pitch.draw(ax=ax)
            
            # Highlight defensive zone
            if pressing_zone == 'high_press':
                color, x_range = 'red', (70, 105)
            elif pressing_zone == 'medium_press':
                color, x_range = 'yellow', (45, 75)
            else:
                color, x_range = 'blue', (0, 40)
            
            rect = plt.Rectangle((x_range[0], 0), x_range[1] - x_range[0], 68, 
                                alpha=0.2, facecolor=color)
            ax.add_patch(rect)
            ax.set_title(f"Defensive Shape - {defensive_shape}", fontsize=12, fontweight='bold')
            
            fig.savefig(self.output_dir / f"{safe_name}_defensive_shape.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            output_files['defensive_shape'] = self.output_dir / f"{safe_name}_defensive_shape.png"
            logger.info(f"Generated defensive shape: {output_files['defensive_shape']}")
            
            # 4. Create attacking patterns visualization
            attacking = tactical_data.get('attacking_patterns', 'Unknown')
            fig, ax = plt.subplots(figsize=(12, 8))
            pitch = Pitch(pitch_type='statsbomb', pitch_color='grass')
            pitch.draw(ax=ax)
            
            # Show attacking pattern info
            ax.text(50, 50, attacking, ha='center', va='center', fontsize=14, wrap=True,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
            ax.set_title(f"Attacking Patterns", fontsize=12, fontweight='bold')
            
            fig.savefig(self.output_dir / f"{safe_name}_attacking_patterns.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            output_files['attacking_patterns'] = self.output_dir / f"{safe_name}_attacking_patterns.png"
            logger.info(f"Generated attacking patterns: {output_files['attacking_patterns']}")
            
            # 5. Generate animations if requested
            if generate_animations:
                # Attacking sequence animation
                frames = self.simulator.simulate_attacking_sequence(tactical_data, num_frames=20)
                fig = self.simulator.create_animation(
                    frames,
                    title=f"Attacking Sequence - {video_name}",
                    save_path=self.output_dir / f"{safe_name}_attacking_animation.gif",
                    fps=5
                )
                plt.close(fig)
                output_files['attacking_animation'] = self.output_dir / f"{safe_name}_attacking_animation.gif"
                logger.info(f"Generated attacking animation: {output_files['attacking_animation']}")
                
                # Defensive shape animation
                frames = self.simulator.simulate_defensive_shape(pressing_zone, formation, num_frames=15)
                fig = self.simulator.create_animation(
                    frames,
                    title=f"Defensive Shape - {video_name}",
                    save_path=self.output_dir / f"{safe_name}_defensive_animation.gif",
                    fps=5
                )
                plt.close(fig)
                output_files['defensive_animation'] = self.output_dir / f"{safe_name}_defensive_animation.gif"
                logger.info(f"Generated defensive animation: {output_files['defensive_animation']}")
        
        except Exception as e:
            logger.error(f"Error processing {video_name}: {e}", exc_info=True)
        
        return output_files
    
    def process_match_comparison(
        self,
        home_data: Dict,
        away_data: Dict,
        match_name: str
    ) -> Dict[str, Path]:
        """
        Process a complete match with both teams' tactical data.
        
        Args:
            home_data: Home team tactical analysis
            away_data: Away team tactical analysis
            match_name: Match identifier for file naming
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        output_files = {}
        safe_name = match_name.replace(' ', '_').replace(':', '')
        
        try:
            # 1. Formation comparison
            home_formation = home_data.get('formation', '4-3-3')
            away_formation = away_data.get('formation', '4-3-3')
            
            fig, axes = self.visualizer.create_formation_comparison(
                home_formation=home_formation,
                away_formation=away_formation,
                title=f"Match Formation Comparison - {match_name}",
                save_path=self.output_dir / f"{safe_name}_formation_comparison.png"
            )
            plt.close(fig)
            output_files['formation_comparison'] = self.output_dir / f"{safe_name}_formation_comparison.png"
            logger.info(f"Generated formation comparison: {output_files['formation_comparison']}")
            
            # 2. Tactical comparison
            fig, axes = self.simulator.create_tactical_comparison(
                home_data=home_data,
                away_data=away_data,
                save_path=self.output_dir / f"{safe_name}_tactical_comparison.png"
            )
            plt.close(fig)
            output_files['tactical_comparison'] = self.output_dir / f"{safe_name}_tactical_comparison.png"
            logger.info(f"Generated tactical comparison: {output_files['tactical_comparison']}")
            
            # 3. Head-to-head analysis
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.axis('off')
            
            # Create comparison table
            comparison_text = "TACTICAL COMPARISON\n" + "="*50 + "\n\n"
            
            aspects = [
                ('Formation', 'formation'),
                ('Build-up Style', 'build_up_style'),
                ('Pressing Style', 'pressing_style'),
                ('Attacking Pattern', 'attacking_patterns'),
                ('Transition Style', 'transition_style'),
            ]
            
            for label, key in aspects:
                home_val = home_data.get(key, 'N/A')
                away_val = away_data.get(key, 'N/A')
                comparison_text += f"{label}:\n"
                comparison_text += f"  Home: {home_val}\n"
                comparison_text += f"  Away: {away_val}\n\n"
            
            ax.text(0.05, 0.95, comparison_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            fig.savefig(self.output_dir / f"{safe_name}_head_to_head.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            output_files['head_to_head'] = self.output_dir / f"{safe_name}_head_to_head.png"
            logger.info(f"Generated head-to-head analysis: {output_files['head_to_head']}")
        
        except Exception as e:
            logger.error(f"Error processing match comparison {match_name}: {e}", exc_info=True)
        
        return output_files
    
    def generate_report(
        self,
        all_visualizations: Dict[str, Dict[str, Path]],
        report_path: Optional[Path] = None
    ) -> Path:
        """
        Generate an HTML report with all visualizations.
        
        Args:
            all_visualizations: Nested dict of video_name -> visualization_dict
            report_path: Path to save the HTML report
            
        Returns:
            Path to the generated report
        """
        report_path = report_path or self.output_dir / "tactical_report.html"
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Football Tactical Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                h1 { color: #333; text-align: center; }
                .video-section { background-color: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .video-title { color: #1f77b4; font-size: 18px; font-weight: bold; margin-bottom: 15px; }
                .visualization-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
                img { max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
                .img-label { font-size: 12px; color: #666; margin-top: 8px; text-align: center; }
            </style>
        </head>
        <body>
            <h1>⚽ Football Tactical Analysis Report</h1>
            <p style="text-align: center; color: #666;">Generated using mplsoccer and tactical analysis</p>
        """
        
        for video_name, visualizations in all_visualizations.items():
            html_content += f"""
            <div class="video-section">
                <div class="video-title">{video_name}</div>
                <div class="visualization-grid">
            """
            
            for viz_type, file_path in visualizations.items():
                if file_path.exists():
                    rel_path = file_path.relative_to(self.output_dir)
                    html_content += f"""
                    <div>
                        <img src="{rel_path}" alt="{viz_type}">
                        <div class="img-label">{viz_type.replace('_', ' ').title()}</div>
                    </div>
                    """
            
            html_content += """
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        report_path.write_text(html_content)
        logger.info(f"Generated HTML report: {report_path}")
        
        return report_path
