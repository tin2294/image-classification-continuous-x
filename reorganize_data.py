import os
import shutil

def organize_data(directory):
    """
    Organizes images in the given directory into subdirectories for each class.
    """
    classes = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
               "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
               "Vegetable/Fruit"]

    os.chdir(directory)

    for i in range(len(classes)):
        class_dir = f"class_{i:02d}"
        
        try:
            os.mkdir(class_dir)
            print(f"Created directory: {class_dir}")
        except FileExistsError:
            print(f"{class_dir} already exists, skipping creation.")
        print(os.listdir(directory))
        files = [f for f in os.listdir(directory) if f.startswith(f"{i}_")]

        for f in files:
            src_path = os.path.join(directory, f)
            dest_path = os.path.join(class_dir, f)
            shutil.move(src_path, dest_path)
            print(f"Moved: {src_path} to {dest_path}")

    print("Data organization complete!")

organize_data("/Users/tingtingchenzeng/image-classification-continuous-x/data/Food-11/training")
organize_data("/Users/tingtingchenzeng/image-classification-continuous-x/data/Food-11/validation")
organize_data("/Users/tingtingchenzeng/image-classification-continuous-x/data/Food-11/evaluation")