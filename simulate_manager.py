import subprocess

cmd = []

for i in range(len(cmd)):
    try:
        print("START "+str(i))
        result = subprocess.run(cmd[i], check=True)
        print(f"Command finished with return code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Command '{e.cmd}' failed with return code {e.returncode}")