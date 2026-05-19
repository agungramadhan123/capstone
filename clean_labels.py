import os
import glob

def clean_labels(directory, max_class_id):
    label_files = glob.glob(os.path.join(directory, "**", "*.txt"), recursive=True)
    count_fixed = 0
    count_lines_removed = 0
    
    for file_path in label_files:
        with open(file_path, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        fixed = False
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
                if class_id <= max_class_id:
                    new_lines.append(line)
                else:
                    fixed = True
                    count_lines_removed += 1
            except ValueError:
                new_lines.append(line)
                
        if fixed:
            with open(file_path, "w") as f:
                f.writelines(new_lines)
            count_fixed += 1

    print(f"Directory {directory}: Fixed {count_fixed} files, removed {count_lines_removed} invalid bounding boxes.")

if __name__ == "__main__":
    clean_labels(r"D:\Semester 6\Capstone\valid\labels", 4)
    clean_labels(r"D:\Semester 6\Capstone\data bubat barat\train\labels", 4)
    clean_labels(r"D:\Semester 6\Capstone\data bubat timur\train\labels", 4)
