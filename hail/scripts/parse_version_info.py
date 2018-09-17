import sys
import configparser


properties_file = sys.argv[1]
output_file = sys.argv[2]

assert output_file.endswith('.py'), output_file

config = configparser.ConfigParser()
files_read = config.read(properties_file)
assert len(files_read) == 1

m = config['Build Metadata']
version_str = f"{m['hailVersion']}-{m['revision'][:12]}"

with open(output_file, 'w') as writer:
	writer.write(f"hail_version = {repr(version_str)}\n")
