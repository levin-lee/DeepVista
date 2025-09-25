prompt1 = """
Task 1: Build the Basic AI Chatbot 
Objective: Create a basic chatbot application with a Python backend and AI-powered responses
Requirements below:
1. Backend
	use the openAI SDK, ensure the short memory in one session

2. For frontend, please use the gradio framework, and support
	Basic chat interface
	Text input field
	Message display area
	Simple, clean interface
	can add conversation

3. Technical Specifications:
	Implement a single-turn conversation
	Basic error handling
	Minimal UI complexity

4. Evaluation Criteria:
	Basic AI response generation
	Code quality
	Ease of setup
"""

prompt2 = """
Added some new features:
1. Added a user clear button to clear the current session
2. Users can add new sessions on the page
3. Beautified the front-end page
"""

prompt3 = """"
The following features have been modified:
The current session switch has been changed to display all sessions on the far right, allowing users to switch to another session by clicking.
The new session function has been moved to the top right corner, indicated by a "+ Add Session" icon.
The "Send" function has been moved down, parallel to the input, and the "Clear Current Session" function has been moved above "Send."
"""