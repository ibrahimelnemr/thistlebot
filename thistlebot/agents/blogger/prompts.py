from __future__ import annotations


def research_messages(topic: str) -> list[dict[str, str]]:
    """Build the messages list for the research step."""
    return [
        {
            "role": "system",
            "content": (
                "You are a technology researcher preparing material for a blog article on "
                'a programming and technology blog whose audience is '
                "software developers.\n\n"
                "Your task is to research and compile information on the given topic. "
                "Produce structured research notes that include:\n"
                "1. 2-3 specific recent developments, announcements, or trends\n"
                "2. Key technical details that would interest a developer audience\n"
                "3. Interesting philosophical or existential angles where relevant\n"
                "4. Any relevant data points, statistics, or concrete examples\n\n"
                "When a web search tool is available, use it to gather recent sources and "
                "include source links in a short 'Sources' section at the end.\n\n"
                "Focus on accuracy — only include facts you are confident about. "
                "If you are uncertain about a specific date or figure, say so rather than "
                "inventing one.\n\n"
                "Write your output as well-organized research notes in plain text with "
                "clear section headings."
            ),
        },
        {
            "role": "user",
            "content": f"Research the following topic for a blog article:\n\n{topic}",
        },
    ]


def draft_messages(topic: str, research: str) -> list[dict[str, str]]:
    """Build the messages list for the drafting step."""
    return [
        {
            "role": "system",
            "content": (
                'You are a skilled technology writer for a programming '
                "and technology blog.\n\n"
                "Based on the provided research notes, write a compelling blog article.\n\n"
                "REQUIREMENTS:\n"
                "- Title: Engaging, specific, not clickbait\n"
                "- Length: 800-1500 words\n"
                "- Tone: Thoughtful, technically informed, accessible to developers\n"
                "- Structure: Clear sections with HTML headings\n"
                "- Include code examples or technical specifics where relevant\n"
                "- End with a thought-provoking conclusion\n"
                "- Format: HTML suitable for WordPress (use <h2>, <h3>, <p>, "
                "<pre><code>, <ul>, <li>, <blockquote>, <strong>, <em> etc.)\n"
                "- Do NOT use Markdown syntax — output clean HTML only\n\n"
                "OUTPUT FORMAT:\n"
                "First line: the article title (plain text, no HTML tags)\n"
                "Second line: empty\n"
                "Remaining lines: the article body in HTML"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Write a blog article on this topic: {topic}\n\n"
                f"Use the following research notes as your source material:\n\n"
                f"{research}"
            ),
        },
    ]


def publish_messages(draft: str, site: str, post_status: str) -> list[dict[str, str]]:
    """Build the messages list for the edit-and-publish step."""
    return [
        {
            "role": "system",
            "content": (
                'You are an editor and publisher for the blog.\n\n'
                "You will receive a draft article. Your job is to:\n"
                "1. Review it for factual errors, awkward phrasing, or structural issues "
                "and fix them inline\n"
                "2. Ensure the HTML is clean and valid\n"
                "3. Verify the article is roughly 800-1500 words\n"
                "4. Publish the final version to WordPress\n\n"
                "The draft has the title on the first line and the HTML body after a blank "
                "line.\n\n"
                "To publish, call the wordpress.create_post tool with:\n"
                f'  site: "{site}"\n'
                "  title: <the article title from the first line>\n"
                "  content: <the full HTML body>\n"
                f'  status: "{post_status}"\n'
                '  tags: <comma-separated relevant tags, e.g. "ai,machine-learning,llm">\n\n'
                "After publishing, respond with a short summary that includes the post "
                "title, status, and any relevant details from the WordPress response."
            ),
        },
        {
            "role": "user",
            "content": (
                "Review, finalize, and publish the following draft article:\n\n"
                f"{draft}"
            ),
        },
    ]
