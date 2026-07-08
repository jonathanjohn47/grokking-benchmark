class Parent:
    def __init__(parent, name):
        parent.name = name
        print("Parent class initialized")
        print(f"Name: {parent.name}")
        
        
class Child(Parent):
    def __init__(child, parent_name, child_name):
        super().__init__(parent_name)
        child.name = child_name
        print("Child class initialized")
        print(f"Name: {child.name}")
        
        
if __name__ == "__main__":
    child_instance1 = Child("Alice", "Bob")
    child_instance2 = Child("Charlie", "David")