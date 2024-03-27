import os
import re
import sys

'''
Printing Logger for convient output viewer
'''

class PrintLogger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()  # Flush the terminal output
        with open(self.log_file, 'a') as f:
            f.write(message)

    def flush(self):
        self.terminal.flush()
        
        
def extract_data_from_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()


        pattern = r'(?P<name>.+?)\s*\|\s*Mean:\s*(?P<mean>\d+\.\d+),\s*Low:\s*\d+\.\d+,\s*High:\s*\d+\.\d+,\s*Std:\s*(?P<std>\d+\.\d+)'
        matches = list(re.finditer(pattern, content))

        if not matches or len(matches) < 5:
            return None

        matches = matches[-5:]

        result = {}
        for match in matches:
            result[match.group('name').strip()] = (match.group('mean'), match.group('std'))

        return result


def main():
    folder_path = './~~'  # Change to your logs folder path
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    log_file = os.path.join(folder_path, '~s')
    logger = PrintLogger(log_file)
    sys.stdout = logger

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            extracted_data = extract_data_from_file(file_path)
            if extracted_data:
                print(f"Data from {file}:")
                for name, values in extracted_data.items():
                    print(f"{values[0]}")
                    print(f"({values[1]})")
                print('----------------------------')


if __name__ == "__main__":
    main()