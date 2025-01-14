# Batch content for running app.py
bat_content_app = '''@echo off
call myenv\\Scripts\\activate
@python.exe app.py %*
@pause
'''

# Save the content to run_app.bat
with open('run_app.bat', 'w') as bat_file:
    bat_file.write(bat_content_app)

print("The 'run_app.bat' file has been created.")
