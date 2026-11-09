
file_path = input("Enter the full path to your attribute definition file: ").strip().strip('"')

with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

print("Total lines:", len(lines))
print("\nFirst 35 lines:")
for i, line in enumerate(lines[:35]):
    print(f"{i}: {line.rstrip()}")
