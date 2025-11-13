# Knowledge Graph Builder – Milestone 4 (Docker)

This is your Milestone 4 project for the Knowledge Graph Builder.
It includes:
- Flask backend (authentication, Neo4j, feedback, admin dashboard)
- Streamlit frontend (upload, semantic search, visualization)
- Docker setup (to run both together easily)

---

## 🚀 How to Run the Project

### 1️⃣ Build the Docker image
Type this command in your terminal (inside your project folder):

```
docker build -t knowmap-app .
```

This means:
> “Build a Docker image named `knowmap-app` using the current folder.”

---

### 2️⃣ Run the container
After the build finishes, run this command:

```
docker run -p 8501:8501 knowmap-app
```

This means:
> “Run the app inside Docker and make it available on port 8501 (your browser).”

---

### 3️⃣ Open the app in your browser
Now open:
👉 [http://localhost:8501](http://localhost:8501)

You’ll see your Streamlit app (frontend) working, which talks to Flask (backend) automatically inside Docker.

---

### 🧑‍💻 Admin Access
The first user who signs up automatically becomes **admin** (user ID = 1).  
Admins can:
- View all feedback
- Manage triples (add/edit/delete)
- View dashboard statistics
