import sys

# replacement strings
WINDOWS_LINE_ENDING = b'\r\n'
UNIX_LINE_ENDING = b'\n'

# relative or absolute file path, e.g.:
print(sys.argv)
try:
    file_path = sys.argv[1]
except:
    print('temp.py file_path')
    exit(0)


with open(file_path, 'rb') as open_file:
    content = open_file.read()

# Windows -> Unix
content = content.replace(WINDOWS_LINE_ENDING, UNIX_LINE_ENDING)

# Unix -> Windows
# content = content.replace(UNIX_LINE_ENDING, WINDOWS_LINE_ENDING)

with open(file_path, 'wb') as open_file:
    open_file.write(content)
