from typing import List, Dict, Optional, Any



def greet(name: str, age: int) -> str:
    return f"Hello {name}, you are {age} years old"



def sum_positive(numbers: List[int]) -> int:
    """Trả về tổng các số dương."""
    return sum(n for n in numbers if n > 0)


def build_user_index(users: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Tạo index user theo email."""
    index: Dict[str, Dict[str, Any]] = {}
    for user in users:
        email = user.get("email")
        if email:
            index[email] = user
    return index


def find_user_age(email: str, user_index: Dict[str, Dict[str, Any]]) -> Optional[int]:
    """Trả về tuổi nếu tìm thấy; None nếu không có."""
    user = user_index.get(email)
    if user is None:
        return None
    age = user.get("age")
    return int(age) if age is not None else None


from dataclasses import dataclass


@dataclass
class User:
    name: str
    age: int
    email: str
    is_active: bool = True  # default value


@dataclass
class Product:
    name: str
    price: float
    quantity: int

    def total_value(self) -> float:
        return self.price * self.quantity




if __name__ == "__main__":
    print("=== Test greet ===")
    print(greet("Andy", 30))

    print("\n=== Test sum_positive ===")
    print(sum_positive([1, -2, 3, 4, -5]))

    print("\n=== Test user index ===")
    users = [
        {"name": "An", "age": 30, "email": "an@example.com"},
        {"name": "Binh", "age": 28, "email": "binh@example.com"},
    ]
    index = build_user_index(users)
    print(index)

    print("\n=== Test find_user_age ===")
    print(find_user_age("an@example.com", index))
    print(find_user_age("unknown@example.com", index))

    print("\n=== Test dataclass User ===")
    user = User(name="An", age=30, email="an@example.com")
    print(user)
    print(user.name, user.age, user.email, user.is_active)

    print("\n=== Test dataclass Product ===")
    p = Product(name="Laptop", price=25_000_000, quantity=2)
    print(p)
    print("Total value:", p.total_value())