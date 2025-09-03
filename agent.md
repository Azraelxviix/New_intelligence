



############   section 1 do not alter this ##########################
user using speech to text so if it isn't clear what he says then try to find the closest next thing or what it would sound like if you sounded it out 
For example if you said stray cats it sounds a lot like stay back so sometimes you might have to phonetically sound what he meant because speech to text 
tries to make up for that and sometimes it uses the wrong words .


You will be mostly autonomous That means you will use the Visual Studio Code command to trigger the elements which are similar the command or key bindings

2.   you will actively solve errors and error logs, you will keep the error logs in gcloud open for yourself to see the problems and solve them. 

don't just talk about a problem, When you find an error you will in a few words or sentences say what that error is and then you will have ready the prompt to fix that error meaning that you're always ready to move forward You don't really stop If you do stop it's just because there's a screen or a prompt on my on my on the user end that I have to press yes or OK so then the next your dialog should always end with the execution of the next command 
so that we can always roll forward 

They are known as Visual Studio Code commands.
These are built-in commands provided via VS Code's command API (triggered through elements like the command palette or keybindings).
When prompted and confirmed (by clicking OK), the corresponding command (e.g., "Chat: Run Prompt...") executes automatically in the VS Code environment.


...existing content...

## Interaction Preference

I prefer to interact with the system via automated VS Code commands. When prompted, I click OK to execute the command, ensuring a smooth and consistent workflow.

...existing content...



---  you will be forward thinking forward planning and "10 steps ahead with commands ready to execute via api vs code trigger 



4.   KEEP EXPLANATIONS BRIEF 1-2 LINES.  SIMPLIFIED.  


5.  ONE EXPLANATION NOT A WHOLE PAGE AT A TIME.



Example: Google Cloud Authentication
When you approve a command, it is executed automatically via the VS Code command API. For example, the following steps occur when you log in to Google Cloud:

The system issues the command:

gcloud auth login

Upon your approval (clicking OK), the command is sent to the integrated terminal using the VS Code API, as demonstrated by:

vscode.commands.executeCommand('workbench.action.terminal.sendSequence', { 
    text: 'gcloud auth login\n' 
});

The terminal then executes the command, opening a browser for authentication and returning a successful login message.



Read agent.md

Summarized conversation history

Here is the current content of your agent.md (first 200 lines):

############ section 1 do not alter this ##########################