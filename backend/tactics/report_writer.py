"""
Renders the coach report JSON (see prompt.py's schema) into a clean
Markdown document. Match-analysis only -- no training/drill sections.
"""


def _bullets(items: list[str]) -> str:
    if not items:
        return "- _None noted._\n"
    return "".join(f"- {item}\n" for item in items)


def _team_section(name: str, team: dict) -> str:
    lines = [f"## {name} Analysis\n"]
    lines.append(f"**Playing style:** {team.get('playing_style', '—')}\n")

    lines.append("\n### Offensive Strengths\n")
    lines.append(_bullets(team.get("offensive_strengths", [])))

    lines.append("\n### Offensive Weaknesses\n")
    lines.append(_bullets(team.get("offensive_weaknesses", [])))

    lines.append("\n### Defensive Strengths\n")
    lines.append(_bullets(team.get("defensive_strengths", [])))

    lines.append("\n### Defensive Weaknesses\n")
    lines.append(_bullets(team.get("defensive_weaknesses", [])))

    lines.append(f"\n**Transition play:** {team.get('transition_play', '—')}\n")

    lines.append("\n### Notable Moments\n")
    lines.append(_bullets(team.get("notable_moments", [])))

    return "".join(lines)


def render_markdown(report: dict, video_label: str = "") -> str:
    overview = report.get("match_overview", {})
    lines = []

    title = f"# Match Tactical Analysis"
    if video_label:
        title += f" — {video_label}"
    lines.append(title + "\n")

    lines.append("## Match Overview\n")
    lines.append(f"- **Duration:** {overview.get('duration_seconds', '—')} seconds")
    lines.append(f"- **Possession — Team A:** {overview.get('possession_team_a', '—')}")
    lines.append(f"- **Possession — Team B:** {overview.get('possession_team_b', '—')}")
    lines.append(f"- **Estimated Formation — Team A:** {overview.get('estimated_formation_team_a', '—')}")
    lines.append(f"- **Estimated Formation — Team B:** {overview.get('estimated_formation_team_b', '—')}")
    lines.append(f"- **Overall Intensity:** {overview.get('overall_intensity', '—')}\n")

    lines.append("## Match Summary\n")
    lines.append(report.get("match_summary", "") + "\n")

    lines.append(_team_section("Team A", report.get("team_a_analysis", {})))
    lines.append(_team_section("Team B", report.get("team_b_analysis", {})))

    lines.append("## Key Match Moments\n")
    moments = report.get("key_match_moments", [])
    if not moments:
        lines.append("_None noted._\n")
    else:
        for m in moments:
            t = m.get("time_seconds", "?")
            team = m.get("team", "?")
            desc = m.get("description", "")
            lines.append(f"- **{t}s (Team {team}):** {desc}")
    lines.append("")

    lines.append("## Player Observations\n")
    players = report.get("player_observations", [])
    if not players:
        lines.append("_None noted._\n")
    else:
        for p in players:
            ref = p.get("reference", "?")
            team = p.get("team", "?")
            obs = p.get("observation", "")
            lines.append(f"- **{ref} (Team {team}):** {obs}")
    lines.append("")

    verdict = report.get("final_verdict", {})
    lines.append("## Final Verdict\n")
    lines.append(f"**Team A:** {verdict.get('team_a_summary', '—')}\n")
    lines.append(f"**Team B:** {verdict.get('team_b_summary', '—')}\n")
    lines.append(f"\n**Overall narrative:** {verdict.get('overall_match_narrative', '—')}\n")

    return "\n".join(lines)
