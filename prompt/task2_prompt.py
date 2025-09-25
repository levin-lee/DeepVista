prompt1 = """
Task 2: add the task2_app.py based on the task1_app.py, and modify tyhe task2._app.py
Objective: Implement YouTube video content processing in the chatbot.

Scenario 1: Linking a YouTube Video
User copies a YouTube video URL (e.g., https://www.youtube.com/watch?v=example)
Pastes the link into the chatbot interface
Chatbot automatically:
Validates the YouTube URL
Retrieves the full video transcript
Generates a concise 3-4 sentence summary of the video content
Displays the summary in the chat interface

Scenario 2: Asking Questions About the Video
After the video summary is displayed, the user can ask specific questions
Examples of potential user queries:
"What are the main points of this video?"
"Can you explain the key concept discussed at 5:30?"
"Summarize the scientific findings in this research presentation"
Chatbot responds by:
Analyzing the full video transcript
Providing a precise, context-aware answer
Citing relevant sections of the transcript if applicable

Requirements:
URL Detection: Automatically detect YouTube URLs in user input
Transcript Extraction: Retrieve video transcripts using appropriate APIs
Content Processing: Generate summaries and enable Q&A based on video content
UI Integration: Display video information and transcript in the chat interface

Technical Considerations:
YouTube API integration or alternative transcript services
Error handling for unavailable transcripts
Rate limiting and caching strategies
Backend API design
"""

prompt1 = """for task2_app.py request as follows:
1. Please do not display transcript on the page.
2. Display video information, such as title, channel, and cover, in the dialog box, not outside of it.
3. Appropriately beautify the chat bot."""