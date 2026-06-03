import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import SessionLocal, engine
import models, auth

models.Base.metadata.create_all(bind=engine)
db = SessionLocal()

users = [
    {'role': 'receptionist', 'email': 'reception@dermaflow.ma', 'password': 'reception123', 'first_name': 'Sara', 'last_name': 'Alami'},
    {'role': 'admin', 'email': 'admin@dermaflow.ma', 'password': 'admin123', 'first_name': 'Mohamed', 'last_name': 'Admin'},
    {'role': 'infermier', 'email': 'infirmier@dermaflow.ma', 'password': 'infirmier123', 'first_name': 'Fatima', 'last_name': 'Zahra'},
]

for u in users:
    existing = db.query(models.User).filter(models.User.email == u['email']).first()
    if not existing:
        user = models.User(
            email=u['email'],
            password_hash=auth.hash_password(u['password']),
            role=u['role'],
            first_name=u['first_name'],
            last_name=u['last_name'],
        )
        db.add(user)
        db.commit()
        print(f"Created: {u['email']}")
    else:
        print(f"Already exists: {u['email']}")

db.close()
print("Done!")