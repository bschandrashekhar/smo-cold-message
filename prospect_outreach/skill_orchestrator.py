"""
Skill Orchestrator: Reads .claude/skills definitions and routes natural language requests to skill implementations.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from . import prospect_research


class SkillOrchestrator:
    """Orchestrates skill execution based on .claude/skills definitions."""

    def __init__(self, skills_dir: Path = None):
        if skills_dir is None:
            # Default to .claude/skills relative to this file
            current_dir = Path(__file__).parent
            skills_dir = current_dir.parent / ".claude" / "skills"

        self.skills_dir = skills_dir
        self.skill_definitions = {}
        self.intent_patterns = {}

        self._load_skill_definitions()
        self._build_intent_patterns()

    def _load_skill_definitions(self):
        """Load skill definitions from .claude/skills/*.md files."""
        if not self.skills_dir.exists():
            return

        for skill_file in self.skills_dir.glob("*-SKILL.md"):
            try:
                content = skill_file.read_text(encoding='utf-8')
                skill_def = self._parse_skill_definition(content)
                if skill_def:
                    skill_name = skill_def['name']
                    self.skill_definitions[skill_name] = skill_def
            except Exception as e:
                print(f"Error loading skill {skill_file}: {e}")

    def _parse_skill_definition(self, content: str) -> Optional[Dict]:
        """Parse a skill definition from markdown content."""
        lines = content.split('\n')
        skill_def = {}

        # Extract name
        name_match = re.search(r'^name:\s*(.+)$', content, re.MULTILINE)
        if name_match:
            skill_def['name'] = name_match.group(1).strip()

        # Extract description
        desc_match = re.search(r'description:\s*>\s*\n(.*?)(?=\n\n|\n---|\Z)', content, re.DOTALL)
        if desc_match:
            skill_def['description'] = desc_match.group(1).strip()

        # Extract inputs section
        inputs_match = re.search(r'## Inputs\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
        if inputs_match:
            skill_def['inputs'] = self._parse_inputs_section(inputs_match.group(1))

        return skill_def if skill_def.get('name') else None

    def _parse_inputs_section(self, inputs_text: str) -> Dict:
        """Parse the inputs section of a skill definition."""
        inputs = {}
        lines = inputs_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('|') and '|' in line:
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 3:
                    field_name = parts[0]
                    field_type = parts[1]
                    description = parts[2]
                    inputs[field_name] = {
                        'type': field_type,
                        'description': description
                    }

        return inputs

    def _build_intent_patterns(self):
        """Build regex patterns for intent recognition."""
        self.intent_patterns = {
            'company-research': [
                r'research\s+(?:this\s+)?(?:company\s+)?([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*?)\s+(https?://[^\s]+|www\.[^\s]+|[a-z0-9-]+\.[a-z]{2,})',
                r'research\s+(https?://[^\s]+|www\.[^\s]+|[a-z0-9-]+\.[a-z]{2,})',
                r'analyze\s+(?:company\s+)?([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*?)\s+(https?://[^\s]+|www\.[^\s]+|[a-z0-9-]+\.[a-z]{2,})',
            ],
            'prospect-research': [
                r'research\s+(?:this\s+)?(?:person\s+|prospect\s+)?([A-Za-z\s]+?)(?:\s+working\s+at|\s+at|\s+employed\s+at)\s+([A-Za-z0-9\s.-]+)',
                r'research\s+(?:prospect\s+)?([A-Za-z\s]+?)(?:\s+at|\s+)([A-Za-z0-9\s]+)',
                r'analyze\s+(?:prospect\s+)?([A-Za-z\s]+?)(?:\s+at|\s+)([A-Za-z0-9\s]+)',
            ]
        }

    def parse_intent(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Parse user input to determine skill intent and extract parameters."""
        user_input = user_input.lower().strip()

        # Check for explicit person indicators first
        person_indicators = ['person', 'prospect', 'individual', 'executive', 'manager', 'director', 'ceo', 'cto', 'cfo']
        is_person_query = any(indicator in user_input for indicator in person_indicators)

        # Prioritize prospect-research if person indicators are present
        if is_person_query:
            skill_name = 'prospect-research'
            patterns = self.intent_patterns.get(skill_name, [])
        else:
            # Check all skills in order
            for skill_name, patterns in self.intent_patterns.items():
                for pattern in patterns:
                    match = re.search(pattern, user_input, re.IGNORECASE)
                    if match:
                        params = self._extract_parameters(skill_name, match, user_input)
                        if params:
                            return {
                                'skill': skill_name,
                                'params': params,
                                'confidence': 0.9
                            }

        # If person query, try prospect patterns
        if is_person_query:
            patterns = self.intent_patterns.get('prospect-research', [])
            for pattern in patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    params = self._extract_parameters('prospect-research', match, user_input)
                    if params:
                        return {
                            'skill': 'prospect-research',
                            'params': params,
                            'confidence': 0.9
                        }

        return None

    def _extract_parameters(self, skill_name: str, match, full_input: str) -> Optional[Dict]:
        """Extract parameters from regex match based on skill definition."""
        if skill_name == 'company-research':
            groups = match.groups()
            if len(groups) >= 2:
                company_name = groups[0].strip()
                url = groups[1].strip()

                # Clean up URL
                if not url.startswith('http'):
                    url = 'https://' + url

                # If company name looks like a stop word, try to extract from URL domain
                stop_words = ['this', 'company', 'the', 'a', 'an']
                if company_name.lower() in stop_words or len(company_name.split()) == 1:
                    # Extract from URL domain, skipping 'www'
                    domain_match = re.search(r'(?:www\.)?([a-z0-9-]+\.[a-z]{2,})', url)
                    if domain_match:
                        domain = domain_match.group(1)
                        company_name = domain.split('.')[0].title()

                return {
                    'company_name': company_name,
                    'url': url,
                    'provider': 'serper'  # Default provider
                }

        elif skill_name == 'prospect-research':
            groups = match.groups()
            if len(groups) >= 2:
                prospect_name = groups[0].strip()
                company_name = groups[1].strip()

                return {
                    'prospect_name': prospect_name,
                    'company_name': company_name,
                    'designation': '',  # Would need more parsing
                    'company_research': '',  # Would need to be provided or fetched
                    'provider': 'serper'
                }

        return None

    def execute_skill(self, skill_name: str, params: Dict) -> str:
        """Execute a skill with given parameters."""
        try:
            return prospect_research.run_skill(skill_name, **params)
        except Exception as e:
            return json.dumps({
                'error': f'Skill execution failed: {str(e)}',
                'skill': skill_name,
                'params': params
            })

    def process_request(self, user_input: str) -> str:
        """Process a user request: parse intent, extract params, execute skill."""
        intent = self.parse_intent(user_input)

        if not intent:
            return json.dumps({
                'error': 'Could not determine intent from request',
                'available_skills': list(self.skill_definitions.keys()),
                'example_requests': [
                    'research MobilePundits www.mobilepundits.com',
                    'research John Smith at Acme Corp'
                ]
            })

        skill_name = intent['skill']
        if skill_name not in self.skill_definitions:
            return json.dumps({
                'error': f'Skill "{skill_name}" not found',
                'available_skills': list(self.skill_definitions.keys())
            })

        params = intent['params']
        result = self.execute_skill(skill_name, params)

        return result


# Global orchestrator instance
_orchestrator = None

def get_orchestrator() -> SkillOrchestrator:
    """Get the global skill orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = SkillOrchestrator()
    return _orchestrator

def process_skill_request(user_input: str) -> str:
    """Convenience function to process a skill request."""
    return get_orchestrator().process_request(user_input)