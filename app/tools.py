"""OpenAI function-calling tool definitions for Karsa AI Assistant.

Defines the set of tools the LLM can invoke to create entities
(tasks, projects, notes, planner entries) in the Karsa system.

Format follows the OpenAI function-calling spec:
https://platform.openai.com/docs/guides/function-calling
"""

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": (
                "Create a new task in the Karsa system. Use this when the user "
                "explicitly asks to add a task, todo, or action item, or when "
                "their request clearly maps to a single actionable item with "
                "a title."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": (
                            "Short, action-oriented task title "
                            "(e.g. 'Buy groceries', 'Email report to team')."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "Optional longer description with context, steps, "
                            "or acceptance criteria."
                        ),
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["LOW", "MEDIUM", "HIGH", "URGENT"],
                        "default": "MEDIUM",
                        "description": (
                            "Priority level. Use URGENT only for time-critical, "
                            "high-impact work. Default MEDIUM if unsure."
                        ),
                    },
                    "deadline": {
                        "type": "string",
                        "description": (
                            "Due date as ISO 8601 (YYYY-MM-DD) or a relative "
                            "phrase the user used (e.g. 'besok', 'tomorrow', "
                            "'minggu depan', 'next Monday')."
                        ),
                    },
                    "projectName": {
                        "type": "string",
                        "description": (
                            "Name of an existing project to link this task to. "
                            "The backend will fuzzy-match the name to a project."
                        ),
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of tag names (e.g. ['work', 'urgent']). "
                            "Tags are created on demand if they do not exist."
                        ),
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_project",
            "description": (
                "Create a new project in the Karsa system. Use this when the "
                "user wants to group related tasks under a named initiative, "
                "or when they describe a multi-step effort with a clear "
                "umbrella name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": (
                            "Project title (e.g. 'Q3 Marketing Plan', "
                            "'Renovate Kitchen')."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "Optional description of the project's scope, "
                            "goals, or context."
                        ),
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["LOW", "MEDIUM", "HIGH", "URGENT"],
                        "default": "MEDIUM",
                        "description": (
                            "Priority level for the project overall. "
                            "Default MEDIUM if unsure."
                        ),
                    },
                    "deadline": {
                        "type": "string",
                        "description": (
                            "Target completion date as ISO 8601 (YYYY-MM-DD) "
                            "or a relative phrase (e.g. 'akhir bulan', "
                            "'end of quarter')."
                        ),
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_note",
            "description": (
                "Create a new note in the Karsa system. Use this when the user "
                "wants to capture a thought, idea, journal entry, or reference "
                "material that is informational rather than actionable. "
                "Markdown is supported in the content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": (
                            "Note title. Should be concise and descriptive "
                            "(e.g. 'Meeting notes 2026-06-18', 'Reading list')."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Note body. Markdown formatting is supported "
                            "(headings, lists, code blocks, links, etc.)."
                        ),
                    },
                    "projectName": {
                        "type": "string",
                        "description": (
                            "Name of an existing project to associate this "
                            "note with. Fuzzy-matched by the backend."
                        ),
                    },
                    "folderName": {
                        "type": "string",
                        "description": (
                            "Name of an existing folder to place this note in. "
                            "Leave empty to place it at the root."
                        ),
                    },
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_planner_entry",
            "description": (
                "Create a time-blocked entry in the Karsa daily planner. "
                "Use this when the user wants to schedule a specific activity "
                "at a specific time on a specific day, or wants to add a "
                "calendar-style block to their plan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": (
                            "Short title for the time block "
                            "(e.g. 'Deep work', 'Lunch break', 'Gym')."
                        ),
                    },
                    "date": {
                        "type": "string",
                        "description": (
                            "Date for the entry in YYYY-MM-DD format. "
                            "Convert relative dates ('tomorrow', 'besok') to "
                            "the absolute YYYY-MM-DD form before calling."
                        ),
                    },
                    "startTime": {
                        "type": "string",
                        "description": (
                            "Start time in HH:MM 24-hour format "
                            "(e.g. '09:00', '14:30')."
                        ),
                    },
                    "endTime": {
                        "type": "string",
                        "description": (
                            "End time in HH:MM 24-hour format. Must be after "
                            "startTime on the same day."
                        ),
                    },
                    "taskId": {
                        "type": "string",
                        "description": (
                            "Optional ID of an existing task to link this "
                            "planner entry to. Leave empty for standalone "
                            "blocks like breaks or generic activities."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "Optional longer description or notes for this "
                            "time block."
                        ),
                    },
                },
                "required": ["title", "date", "startTime", "endTime"],
            },
        },
    },
]


__all__ = ["TOOLS"]
