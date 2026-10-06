from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'Rishab'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5.0, description='decimal value representing the cgpa of the student')

new_student = {'age': 25, 'email': 'abc@gmail.com'}

student = Student(**new_student)

student_dict = dict(student)
student_json = student.model_dump_json

print(student)
print(student_dict)
print(student_json)