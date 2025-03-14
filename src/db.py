import psycopg2
from psycopg2 import sql

# Database connection configuration
DB_CONFIG = {
    "dbname": "mnist_db",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": "5432"
}

def get_db_connection():
    """Establish a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

if __name__ == "__main__":
    conn = get_db_connection()
    if conn:
        print("Database connection successful!")
        conn.close()
